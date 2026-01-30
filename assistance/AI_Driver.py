import math
from typing import Dict, Any, List, Tuple
import json
import pyautogui

from AI_Control import AIControlState
from assistance.base_system import AssistanceSystem
from core.event_bus import EventBus
from core.settings_manager import SettingsManager
from vehicles.own_vehicle import OwnVehicle
from vehicles.vehicle import Vehicle


def dist(a=(0, 0, 0), b=(0, 0, 0)):
    """Determine the distance between two points."""
    return math.sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]) + (b[2] - a[2]) * (b[2] - a[2]))


def load_routes_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Lädt Routen aus einer Datei"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data.get('roads', [])


def get_closest_index_on_route(carX, carY, carZ, route_points):
    """
    Find the index of the closest point on the route to the car's current position.

    Args:
        carX, carY, carZ: Current car position coordinates
        route_points: Dict containing 'path' key with list of [x, y, z] points

    Returns:
        int: Index of the closest point in the route
    """
    path = route_points.get('path', [])
    if not path:
        return 0

    car_pos = (carX, carY, carZ)
    min_distance = float('inf')
    closest_index = 0

    for i, point in enumerate(path):
        distance = dist(car_pos, tuple(point))
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index


def get_next_points_on_route(current_index, route_points, num_points=5):
    """
    Get the next points on the route, wrapping around if necessary.

    Args:
        current_index: Current position index on the route
        route_points: Dict containing 'path' key with list of [x, y, z] points
        num_points: Number of points to retrieve (default: 5)

    Returns:
        List of next points on the route
    """
    path = route_points.get('path', [])
    if not path:
        return []

    next_points = []
    path_length = len(path)

    for i in range(num_points):
        # Wrap around using modulo
        index = (current_index + i) % path_length
        next_points.append(path[index])

    return next_points


def analyze_upcoming_track(route_points) -> Tuple[float, Tuple[float, float, float]]:
    """
    Analyze the upcoming track section to determine curvature and target steering point.

    Args:
        route_points: List of upcoming points (typically 5 points)

    Returns:
        Tuple containing:
        - average_curvature: Average curvature of the upcoming section
        - target_point: Average position of points 2-3 (indices 1-2) to steer towards
    """
    if len(route_points) < 3:
        # Not enough points to analyze
        return 0.0, tuple(route_points[1] if len(route_points) > 1 else route_points[0])

    # Calculate curvature by analyzing angle changes between consecutive segments
    curvatures = []

    for i in range(len(route_points) - 2):
        p1 = route_points[i]
        p2 = route_points[i + 1]
        p3 = route_points[i + 2]

        # Calculate vectors
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Calculate angles
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])

        # Calculate angle difference (curvature indicator)
        angle_diff = angle2 - angle1

        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Distance between points
        segment_length = dist(p2, p3)

        # Curvature = angle change / distance
        if segment_length > 0:
            curvature = abs(angle_diff) / segment_length
            curvatures.append(curvature)

    # Average curvature
    average_curvature = sum(curvatures) / len(curvatures) if curvatures else 0.0

    # Calculate target point (average of points at indices 1 and 2, skipping point 0)
    if len(route_points) >= 3:
        target_point = (
            (route_points[1][0] + route_points[2][0]) / 2,
            (route_points[1][1] + route_points[2][1]) / 2,
            (route_points[1][2] + route_points[2][2]) / 2
        )
    else:
        target_point = tuple(route_points[1])

    return average_curvature, target_point


class SimplePIDController:
    """
    Simple PID controller for controlling a single value.

    Attributes:
        kp: Proportional gain - how strongly to react to current error
        ki: Integral gain - how strongly to react to accumulated error
        kd: Derivative gain - how strongly to react to rate of change
    """

    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error: float, dt: float = 0.016) -> float:
        """
        Calculate control output based on error.

        Args:
            error: Current error (target - actual)
            dt: Time step (default 0.016 for ~60 FPS)

        Returns:
            Control output value
        """
        # Proportional term
        p_term = self.kp * error

        # Integral term (accumulated error over time)
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term (rate of change of error)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        # Update previous error
        self.prev_error = error

        # Return combined output
        return p_term + i_term + d_term

    def reset(self):
        """Reset the controller state"""
        self.prev_error = 0.0
        self.integral = 0.0



class AIDriver(AssistanceSystem):
    """Ai Driver"""

    def __init__(self, event_bus: EventBus, settings: SettingsManager):
        super().__init__("AIDriver", event_bus, settings)
        self.routes = None
        self.ai_controller = None
        self.event_bus.subscribe("AI_Controller_initialized", self._on_ai_controller_initialized)

        # PID controllers for each vehicle (will be created on demand)
        self.steering_controllers: Dict[int, SimplePIDController] = {}
        self.speed_controllers: Dict[int, SimplePIDController] = {}

        # Tunable parameters - adjust these to change behavior
        self.STEERING_KP = 2.0  # Proportional gain for steering (increase for more aggressive steering)
        self.STEERING_KI = 0.1  # Integral gain for steering (increase to correct persistent offset)
        self.STEERING_KD = 0.5  # Derivative gain for steering (increase to reduce oscillation)

        self.SPEED_KP = 3.0  # Proportional gain for speed (increase for faster acceleration)
        self.SPEED_KI = 0.2  # Integral gain for speed (increase to maintain target speed better)
        self.SPEED_KD = 0.1  # Derivative gain for speed (increase to smooth speed changes)

        # Speed parameters
        self.BASE_SPEED = 40.0  # Base speed in km/h (or your game's unit)
        self.MIN_SPEED = 15.0  # Minimum speed on tight curves
        self.CURVATURE_THRESHOLD = 0.02  # Curvature above which to slow down

    def _on_ai_controller_initialized(self, ai_controller):
        self.ai_controller = ai_controller

    def _get_or_create_controllers(self, vehicle_id: int) -> Tuple[SimplePIDController, SimplePIDController]:
        """Get or create PID controllers for a vehicle"""
        if vehicle_id not in self.steering_controllers:
            self.steering_controllers[vehicle_id] = SimplePIDController(
                self.STEERING_KP, self.STEERING_KI, self.STEERING_KD
            )
        if vehicle_id not in self.speed_controllers:
            self.speed_controllers[vehicle_id] = SimplePIDController(
                self.SPEED_KP, self.SPEED_KI, self.SPEED_KD
            )
        return self.steering_controllers[vehicle_id], self.speed_controllers[vehicle_id]

    def calculate_steering_error(self, vehicle_x: float, vehicle_y: float,
                                 vehicle_heading: float, target_point: Tuple[float, float, float]) -> float:
        """
        Calculate steering error (angle to target point).

        Args:
            vehicle_x, vehicle_y: Current vehicle position
            vehicle_heading: Current vehicle heading in radians
            target_point: Target point to steer towards (x, y, z)

        Returns:
            Steering error in radians (positive = need to turn right, negative = turn left)
        """
        # Vector from vehicle to target
        dx = target_point[0] - vehicle_x
        dy = target_point[1] - vehicle_y

        # Angle to target
        angle_to_target = math.atan2(dy, dx)

        # Calculate error (difference between where we're heading and where we want to go)
        error = angle_to_target - vehicle_heading

        # Normalize to [-pi, pi]
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        return error

    def calculate_target_speed(self, curvature: float) -> float:
        """
        Calculate target speed based on upcoming curvature.

        Args:
            curvature: Average curvature of upcoming section

        Returns:
            Target speed
        """
        if curvature < self.CURVATURE_THRESHOLD:
            # Straight section - use base speed
            return self.BASE_SPEED
        else:
            # Curved section - reduce speed based on curvature
            # Higher curvature = lower speed
            speed_reduction = (curvature - self.CURVATURE_THRESHOLD) * 200.0
            target_speed = max(self.MIN_SPEED, self.BASE_SPEED - speed_reduction)
            return target_speed

    def monitor_ai(self, aii):
        if aii.RPM >5000:
            self.ai_controller.control_ai(aii.PLID, AIControlState(
                shift_up=True,
            ))
        if aii.RPM < 2000 and aii.Gear > 2:
            self.ai_controller.control_ai(aii.PLID, AIControlState(
                shift_down=True,
            ))
        if aii.Gear < 1:
            self.ai_controller.control_ai(aii.PLID, AIControlState(
                shift_up=True,
            ))
        if aii.RPM <1000:
            self.ai_controller.control_ai(aii.PLID, AIControlState(
                ignition=True,
            ))
        print(f"AI Info for Vehicle {aii.PLID}: RPM={aii.RPM}, Gear={aii.Gear}")


    def process(self, own_vehicle: OwnVehicle, vehicles: Dict[int, Vehicle]) -> Dict[str, Any]:
        """Verarbeitet die Auto-Hold-Logik"""
        if self.routes is None:
            self.routes = load_routes_from_file("StreetMapCreator/track_data.json")
            self.routes = {road['road_id']: road for road in self.routes}

        # Initialize test vehicle with route 20
        for vehicle_id in vehicles.keys():
            if vehicle_id == 2:
                if vehicles.get(vehicle_id).current_route is None:
                    vehicles.get(vehicle_id).current_route = 20


                    self.ai_controller.bind_ai_info_handler(2, self.monitor_ai)
                    self.ai_controller.request_ai_info(2, repeat_interval=200)
        # Process each vehicle that has a route assigned
        for vehicle_id in vehicles.keys():
            vehicle = vehicles.get(vehicle_id)

            if vehicle.current_route is not None:
                route_points = self.routes[vehicle.current_route]

                # Get vehicle position (convert from game units)
                vehicle_x = vehicle.data.x / 65536
                vehicle_y = vehicle.data.y / 65536
                vehicle_z = vehicle.data.z / 65536

                # Find closest point and get upcoming points
                closest_index = get_closest_index_on_route(
                    vehicle_x, vehicle_y, vehicle_z, route_points
                )
                next_five_points = get_next_points_on_route(closest_index, route_points)

                # Analyze the upcoming track section
                curvature, target_point = analyze_upcoming_track(next_five_points)

                # Get PID controllers for this vehicle
                steering_controller, speed_controller = self._get_or_create_controllers(vehicle_id)

                # Calculate wanted speed based on curvature
                target_speed = self.calculate_target_speed(curvature)

                # Get current speed (assuming vehicle.data.speed is in appropriate units)
                current_speed = vehicle.data.speed if hasattr(vehicle.data, 'speed') else 0.0

                # Calculate speed error
                speed_error = target_speed - current_speed

                # Get speed control output from PID
                speed_control = speed_controller.update(speed_error)

                # Convert speed control to throttle/brake (0-100 range)
                if speed_control > 0:
                    # Need to accelerate
                    throttle = min(100, max(0, speed_control))
                    brake = 0
                else:
                    # Need to brake
                    throttle = 0
                    brake = min(100, max(0, -speed_control))

                # Calculate steering based on target_point
                # Get vehicle heading (assuming it's available, otherwise use velocity direction)
                if hasattr(vehicle.data, 'heading'):
                    vehicle_heading = vehicle.data.heading
                else:
                    # Fallback: calculate heading from velocity if available
                    # You may need to adjust this based on your Vehicle class structure
                    vehicle_heading = 0.0  # Default, should be replaced with actual heading

                # Calculate steering error
                steering_error = self.calculate_steering_error(
                    vehicle_x, vehicle_y, vehicle_heading, target_point
                )

                # Get steering control output from PID
                steering_control = steering_controller.update(steering_error)

                # Convert to game's steering range (-100 to 100)
                # The multiplier here converts from radians to game units
                # Adjust the 50 multiplier to change steering sensitivity
                steering = max(-100, min(100, steering_control * 50))

                # Send control commands to AI controller
                if self.ai_controller is not None:

                    self.ai_controller.control_ai(vehicle_id, AIControlState(
                        throttle=int(throttle),
                        brake=int(brake),
                        steer=int(steering),
                    ))


                    print(f"Vehicle {vehicle_id}: Curvature={curvature:.4f}, Target={target_point}")
                    print(
                        f"  Speed: {current_speed:.1f} → {target_speed:.1f} | Throttle: {throttle:.0f}% | Brake: {brake:.0f}%")
                    print(f"  Steering Error: {steering_error:.3f} rad | Steer: {steering:.0f}")

        return {
            'ai_active': True
        }