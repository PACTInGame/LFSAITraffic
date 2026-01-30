import pyinsim
from vehicles.vehicle import Vehicle


class OwnVehicle(Vehicle):
    """Repräsentiert das eigene Fahrzeug mit erweiterten Daten"""

    def __init__(self):
        super().__init__(0)  # Player ID wird später gesetzt

        # OutGauge-Daten
        self.fuel: float = 0.0
        self.rpm: int = 0
        self.gear: int = 0
        self.brake: float = 0.0
        self.throttle: float = 0.0
        self.clutch: float = 0.0
        self.turbo: float = 0.0

        # Lichter und Indikatoren
        self.indicator_left: bool = False
        self.indicator_right: bool = False
        self.hazard_lights: bool = False
        self.full_beam_light: bool = False
        self.low_beam_light: bool = False

        # Warnleuchten
        self.tc_light: bool = False
        self.abs_light: bool = False
        self.handbrake_light: bool = False
        self.battery_light: bool = False
        self.oil_light: bool = False
        self.eng_light: bool = False



    def update_outgauge_data(self, packet):
        """Aktualisiert Daten aus OutGauge-Paket"""
        self.data.player_id = packet.PLID
        self.fuel = packet.Fuel
        self.data.speed = packet.Speed * 3.6  # Convert to km/h
        self.rpm = packet.RPM
        self.gear = packet.Gear
        self.brake = packet.Brake
        self.throttle = packet.Throttle
        self.clutch = packet.Clutch
        self.turbo = packet.Turbo

        # Lichter auswerten
        lights = packet.ShowLights
        self.indicator_left = bool(pyinsim.DL_SIGNAL_L & lights)
        self.indicator_right = bool(pyinsim.DL_SIGNAL_R & lights)
        self.hazard_lights = self.indicator_left and self.indicator_right
        self.full_beam_light = bool(pyinsim.DL_FULLBEAM & lights)
        self.low_beam_light = bool(pyinsim.DL_DIPPED & lights)
        self.tc_light = bool(pyinsim.DL_TC & lights)
        self.abs_light = bool(pyinsim.DL_ABS & lights)
        self.handbrake_light = bool(pyinsim.DL_HANDBRAKE & lights)
        self.battery_light = bool(pyinsim.DL_BATTERY & lights)
        self.oil_light = bool(pyinsim.DL_OILWARN & lights)
        self.eng_light = bool(pyinsim.DL_ENGINE & lights)
