import pyinsim
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import IntEnum


class AIControl(IntEnum):
    """AI Control Input Types"""
    STEER = 0
    THROTTLE = 1
    BRAKE = 2
    SHIFT_UP = 3
    SHIFT_DOWN = 4
    IGNITION = 5
    EXTRA_LIGHT = 6
    HEADLIGHTS = 7
    SIREN = 8
    HORN = 9
    FLASH = 10
    CLUTCH = 11
    HANDBRAKE = 12
    INDICATORS = 13
    GEAR = 14
    LOOK = 15
    PIT_SPEED = 16
    TC_DISABLE = 17
    FOG_REAR = 18
    FOG_FRONT = 19

    # Special controls
    SEND_AI_INFO = 240
    REPEAT_AI_INFO = 241
    SET_HELP_FLAGS = 253
    RESET_INPUTS = 254
    STOP_CONTROL = 255


class HeadlightMode(IntEnum):
    """Headlight modes"""
    OFF = 1
    SIDE = 2
    LOW = 3
    HIGH = 4


class IndicatorMode(IntEnum):
    """Indicator modes"""
    CANCEL = 1
    LEFT = 2
    RIGHT = 3
    HAZARD = 4


class LookDirection(IntEnum):
    """Look directions"""
    NONE = 0
    LEFT = 4
    LEFT_PLUS = 5
    RIGHT = 6
    RIGHT_PLUS = 7


class SirenMode(IntEnum):
    """Siren modes"""
    FAST = 1
    SLOW = 2


@dataclass
class AIControlState:
    """
    Represents the complete control state for an AI vehicle.
    All analog values are in range 0-65535 (0-100% is auto-converted).
    """
    # Analog controls (0-65535)
    steer: Optional[int] = None  # 1=hard left, 32768=center, 65535=hard right
    throttle: Optional[int] = None
    brake: Optional[int] = None
    clutch: Optional[int] = None
    handbrake: Optional[int] = None

    # Gear controls
    shift_up: Optional[bool] = None
    shift_down: Optional[bool] = None
    gear: Optional[int] = None  # For H-shifter (255 for sequential)

    # Lights and signals
    ignition: Optional[bool] = None
    headlights: Optional[HeadlightMode] = None
    extra_light: Optional[bool] = None
    indicators: Optional[IndicatorMode] = None
    fog_front: Optional[bool] = None
    fog_rear: Optional[bool] = None
    flash: Optional[bool] = None

    # Audio
    horn: Optional[int] = None  # 1-5
    siren: Optional[SirenMode] = None

    # Other
    look: Optional[LookDirection] = None
    pit_speed_limiter: Optional[bool] = None
    traction_control_disable: Optional[bool] = None


class AICarController:
    """
    Main controller class for managing AI vehicles in Live for Speed.
    Provides high-level interface for controlling AI cars.
    """

    # Constants for value conversion
    ANALOG_MAX = 65535
    ANALOG_CENTER = 32768

    def __init__(self, insim: pyinsim.insim):
        """
        Initialize the AI controller.

        Args:
            insim: Active pyinsim instance
        """
        self.insim = insim
        self._ai_info_handlers: Dict[int, callable] = {}

    def _normalize_analog(self, value: float, center_zero: bool = False) -> int:
        """
        Convert percentage (0-100) or raw value to InSim range.

        Args:
            value: Value to convert (0-100 for percentage, or raw 0-65535)
            center_zero: If True, treats input as -100 to +100 with 0 as center

        Returns:
            Value in range 0-65535
        """
        if center_zero:
            # -100 to +100 range, map to 0-65535
            if -100 <= value <= 100:
                return min(max(1,int(self.ANALOG_CENTER + (value / 100.0) * self.ANALOG_CENTER)), self.ANALOG_MAX)
            return min(max(1, int(value)), self.ANALOG_MAX)
        else:
            # 0-100 percentage or raw value
            if 0 <= value <= 100:
                return min(max(1, int((value / 100.0) * self.ANALOG_MAX)), self.ANALOG_MAX)
            return min(max(1, int(value)), self.ANALOG_MAX)

    def _build_input_list(self, state: AIControlState) -> List[pyinsim.AIInputVal]:
        """
        Convert AIControlState to list of AIInputVal objects.

        Args:
            state: Control state to convert

        Returns:
            List of AIInputVal objects ready to send
        """
        inputs = []

        # Analog controls
        if state.steer is not None:
            print(f"Normalized Steering: {self._normalize_analog(state.steer, center_zero=True)}")
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.STEER,
                Value=self._normalize_analog(state.steer, center_zero=True)
            ))

        if state.throttle is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.THROTTLE,
                Value=self._normalize_analog(state.throttle)
            ))

        if state.brake is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.BRAKE,
                Value=self._normalize_analog(state.brake)
            ))

        if state.clutch is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.CLUTCH,
                Value=self._normalize_analog(state.clutch)
            ))

        if state.handbrake is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.HANDBRAKE,
                Value=self._normalize_analog(state.handbrake)
            ))

        # Gear controls
        if state.shift_up is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.SHIFT_UP,
                Value=1 if state.shift_up else 0
            ))

        if state.shift_down is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.SHIFT_DOWN,
                Value=1 if state.shift_down else 0
            ))

        if state.gear is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.GEAR,
                Value=state.gear
            ))

        # Toggles and modes
        if state.ignition is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.IGNITION,
                Value=1 if state.ignition else 0
            ))

        if state.headlights is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.HEADLIGHTS,
                Value=state.headlights.value
            ))

        if state.extra_light is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.EXTRA_LIGHT,
                Value=1 if state.extra_light else 0
            ))

        if state.indicators is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.INDICATORS,
                Value=state.indicators.value
            ))

        if state.fog_front is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.FOG_FRONT,
                Value=1 if state.fog_front else 0
            ))

        if state.fog_rear is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.FOG_REAR,
                Value=1 if state.fog_rear else 0
            ))

        if state.flash is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.FLASH,
                Value=1 if state.flash else 0
            ))

        # Audio
        if state.horn is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.HORN,
                Value=max(1, min(5, state.horn))
            ))

        if state.siren is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.SIREN,
                Value=state.siren.value
            ))

        # Other
        if state.look is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.LOOK,
                Value=state.look.value
            ))

        if state.pit_speed_limiter is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.PIT_SPEED,
                Value=1 if state.pit_speed_limiter else 0
            ))

        if state.traction_control_disable is not None:
            inputs.append(pyinsim.AIInputVal(
                Input=AIControl.TC_DISABLE,
                Value=1 if state.traction_control_disable else 0
            ))

        return inputs

    def control_ai(self, plid: int, state: AIControlState) -> None:
        """
        Send control commands to an AI vehicle.

        Args:
            plid: Player ID of the AI vehicle
            state: AIControlState with desired controls

        Example:
            controller.control_ai(1, AIControlState(
                throttle=75,
                steer=10,  # Slight right turn
                headlights=HeadlightMode.LOW
            ))
        """
        inputs = self._build_input_list(state)
        if inputs:
            print("Inputs:", inputs)
            self.insim.send(pyinsim.ISP_AIC, PLID=plid, Inputs=inputs)

    def control_ai_raw(self, plid: int, controls: Dict[str, Any]) -> None:
        """
        Send control commands using a dictionary (alternative interface).

        Args:
            plid: Player ID of the AI vehicle
            controls: Dictionary of control name -> value

        Example:
            controller.control_ai_raw(1, {
                'throttle': 50,
                'brake': 0,
                'steer': -20,
                'indicators': IndicatorMode.LEFT
            })
        """
        state = AIControlState(**controls)
        self.control_ai(plid, state)

    def reset_ai_controls(self, plid: int) -> None:
        """
        Reset all AI controls to neutral/off state.

        Args:
            plid: Player ID of the AI vehicle
        """
        inputs = [pyinsim.AIInputVal(Input=AIControl.RESET_INPUTS)]
        self.insim.send(pyinsim.ISP_AIC, PLID=plid, Inputs=inputs)

    def stop_ai_control(self, plid: int) -> None:
        """
        Stop controlling the AI (return to default AI behavior).

        Args:
            plid: Player ID of the AI vehicle
        """
        inputs = [pyinsim.AIInputVal(Input=AIControl.STOP_CONTROL)]
        self.insim.send(pyinsim.ISP_AIC, PLID=plid, Inputs=inputs)

    def request_ai_info(self, plid: int, repeat_interval: Optional[int] = None) -> None:
        """
        Request AI information (RPM, gear, etc.).

        Args:
            plid: Player ID of the AI vehicle
            repeat_interval: If provided, repeat info every N milliseconds (100-60000)

        Example:
            # One-time request
            controller.request_ai_info(1)

            # Continuous updates every 100ms
            controller.request_ai_info(1, repeat_interval=100)
        """
        if repeat_interval is not None:
            inputs = [pyinsim.AIInputVal(
                Input=AIControl.REPEAT_AI_INFO,
                Time=max(100, min(60000, repeat_interval))
            )]
        else:
            inputs = [pyinsim.AIInputVal(Input=AIControl.SEND_AI_INFO)]

        self.insim.send(pyinsim.ISP_AIC, PLID=plid, Inputs=inputs)

    def bind_ai_info_handler(self, plid: int, handler: callable) -> None:
        """
        Bind a callback for AI info updates for specific vehicle.

        Args:
            plid: Player ID to monitor
            handler: Callback function(aii) where aii is ISP_AII packet

        Example:
            def my_handler(aii):
                print(f"RPM: {aii.RPM}, Gear: {aii.Gear}")

            controller.bind_ai_info_handler(1, my_handler)
        """
        self._ai_info_handlers[plid] = handler

        # Set up global handler if not already set
        if not hasattr(self, '_global_handler_set'):
            def global_handler(insim, aii):
                if aii.PLID in self._ai_info_handlers:
                    self._ai_info_handlers[aii.PLID](aii)

            self.insim.bind(pyinsim.ISP_AII, global_handler)
            self._global_handler_set = True


# Convenience functions for common operations
class AIControlHelper:
    """Helper class with common control patterns"""

    @staticmethod
    def drive_forward(speed_percent: float = 100) -> AIControlState:
        """Create state for driving straight forward"""
        return AIControlState(
            throttle=speed_percent,
            steer=0,
            brake=0
        )

    @staticmethod
    def brake_to_stop(brake_percent: float = 100) -> AIControlState:
        """Create state for braking"""
        return AIControlState(
            throttle=0,
            brake=brake_percent,
            steer=0
        )

    @staticmethod
    def turn(steer_angle: float, throttle: float = 50) -> AIControlState:
        """
        Create state for turning.

        Args:
            steer_angle: -100 (hard left) to +100 (hard right)
            throttle: Throttle percentage
        """
        return AIControlState(
            steer=steer_angle,
            throttle=throttle
        )

    @staticmethod
    def stop_and_park() -> AIControlState:
        """Create state for stopped vehicle with parking setup"""
        return AIControlState(
            throttle=0,
            brake=0,
            handbrake=100,
            indicators=IndicatorMode.HAZARD,
            ignition=False
        )


# Example usage
if __name__ == "__main__":
    # Initialize InSim
    insim = pyinsim.insim(b'127.0.0.1', 29999, Admin=b'')

    # Create controller
    controller = AICarController(insim)

    # Example 1: Simple control
    controller.control_ai(2, AIControlState(
        throttle=75, # 75% throttle
        brake=50, # no brake
        steer=10,  # Slight right
    ))

    # Example 2: Using dictionary interface
    controller.control_ai_raw(2, {
        'throttle': 50,
        'steer': -20,  # Left turn
        'indicators': IndicatorMode.LEFT
    })

    # Example 3: Using helper functions
    #controller.control_ai(2, AIControlHelper.drive_forward(speed_percent=80))


    # Example 4: Request AI info with callback
    def monitor_ai(aii):
        print(f"AI {aii.PLID}: RPM={aii.RPM}, Gear={aii.Gear}")


    controller.bind_ai_info_handler(1, monitor_ai)
    controller.request_ai_info(1, repeat_interval=200)

    # Start InSim
    pyinsim.run()