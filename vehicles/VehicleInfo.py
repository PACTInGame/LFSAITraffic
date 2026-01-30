class VehicleInfo:
    def __init__(self):
        self.player_id = 0
        self.vehicle_model = ""
        self.vehicle_size = {
            "width": 0.0,
            "length": 0.0
        }

        self.coordinates = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        }
        self.heading = 0
        self.direction = 0
        self.steer_forces = 0
        self.speed_power_axle = 0
        self.previous_speed_power_axle = 0
        self.speed_over_ground = 0
        self.previous_speed_over_ground = 0
        self.acceleration = 0

        self.accelerator_pedal_pos = 0
        self.brake_pedal_pos = 0
        self.clutch_pedal_pos = 0
        self.gearbox_mode = "manual"  # automatic, manual, sequential
        self.controller_type = "mouse" # mouse, keyboard, wheel

        self.gear = 0
        self.engine_rpm = 0

        self.fuel_capacity = -1
        self.redline_rpm = -1
        self.max_gears = -1

        self.indicator_left = False
        self.indicator_right = False
        self.indicator_hazard = False
        self.auto_clutch = False
        self.battery_light = False
        self.oil_light = False
        self.abs_light = False
        self.handbrake_light = False
        self.tc_light = False
        self.full_beam_light = False
        self.engine_light = False
        self.low_beam_light = False

        self.player_name = ""
        self.cname = ""

        self.eng_type = "ICE"  # ICE, BEV

        self.driving_mode = "driver" # driver, acc, lsa



