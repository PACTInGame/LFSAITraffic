import time
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class VehicleData:
    """Container für Fahrzeugdaten"""
    player_id: int = 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    speed: float = 0.0
    heading: float = 0.0
    direction: float = 0.0
    distance_to_player: float = 0.0
    angle_to_player: float = 0.0
    acceleration: float = 0.0
    cname = "Unknown"
    pname = "Unknown"
    control_mode = 0


class Vehicle:
    """Repräsentiert ein Fahrzeug auf der Strecke"""

    def __init__(self, player_id: int):
        self.data = VehicleData(player_id=player_id)
        self.last_update = 0
        self.previous_speed = 0.0
        self.current_route = None


    def update_position(self, x: float, y: float, z: float, heading: float,
                        direction: float, speed: float):
        """Aktualisiert Position und Bewegungsdaten"""
        self.data.x = x
        self.data.y = y
        self.data.z = z
        self.data.heading = heading
        self.data.direction = direction

        # Berechne Beschleunigung
        self.data.speed = speed
        self.data.acceleration = (speed - self.previous_speed) * 2.778  # Umrechnung von km/h auf m/s²
        self.previous_speed = self.data.speed



    def update_distance_to_player(self, player_x: float, player_y: float, player_z: float):
        """Berechnet Distanz zum Spieler"""
        dx = self.data.x - player_x
        dy = self.data.y - player_y
        dz = self.data.z - player_z
        dx = dx / 65536
        dy = dy / 65536
        dz = dz / 65536
        self.data.distance_to_player = math.sqrt(dx * dx + dy * dy + dz * dz)

    def update_angle_to_player(self, player_x: float, player_y: float, own_heading: float):
        """Berechnet Winkel zum Spieler"""
        ang = (math.atan2((player_x / 65536 - self.data.x / 65536),
                          (player_y / 65536 - self.data.y / 65536)) * 180.0) / 3.1415926535897931
        if ang < 0.0:
            ang = 360.0 + ang
        consider_dir = ang + own_heading / 182
        if consider_dir > 360.0:
            consider_dir -= 360.0
        angle = (consider_dir + 180.0) % 360.0
        self.data.angle_to_player = angle

    def update_model_and_driver(self, cname: str, pname: str, control_mode: int) -> bool:
        """Aktualisiert Modell- und Fahrerdaten"""
        changed = self.data.cname != cname or self.data.pname != pname or self.data.control_mode != control_mode
        self.data.cname = cname
        self.data.pname = pname
        self.data.control_mode = control_mode
        return changed