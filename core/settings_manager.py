import json
import os
from typing import Any, Dict

class SettingsManager:
    """Verwaltet alle Einstellungen mit Persistierung"""

    def __init__(self, settings_file: str = "settings.json"):
        self.settings_file = settings_file
        self._settings: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {
            'forward_collision_warning': True,
            'blind_spot_warning': True,
            'cross_traffic_warning': True,
            'automatic_gearbox': False,
            'collision_warning_distance': 1, # 0 = Early, 1 = Normal, 2 = Late
            'automatic_emergency_brake': 2,  # 0 = Off, 1 = Warn, 2 = Warn & Brake
            'auto_hold': True,
            'adaptive_lights': True,
            'own_control_mode': 0,  # 0 = Mouse, 1 = Keyboard, 2 = Joystick

            'park_distance_control': True,
            'parking_emergency_brake': True,
            'park_distance_control_mode': 0,  # 0 = Off, 1 = Visual, 2 = Visual & Audio

            'language': 'de',
            'unit': "metric",  # 'metric' or 'imperial'

            'ui_refresh_rate': 50,
            'assistance_refresh_rate': 100,
            'hud_height': 119,
            'hud_width': 90,

            'user_handbrake_key': "q",
            'user_shift_up_key': "s",
            'user_shift_down_key': "x",
            'user_clutch_key': "c",
            'user_ignition_key': "i",
            'user_brake_key': "down",

            'user_axis_steering': 8,
            'user_axis_throttle': 9,
            'user_axis_brake': 12,
            'user_axis_clutch': 13,
            'vjoy_axis_1': 15,

        }
        self.load()

    def get(self, key: str, default: Any = None) -> Any:
        """Holt einen Einstellungswert"""
        return self._settings.get(key, default or self._defaults.get(key))

    def set(self, key: str, value: Any):
        """Setzt einen Einstellungswert"""
        print(f"Setting {key} to {value}")
        self._settings[key] = value
        self.save()

    def load(self):
        """Lädt Einstellungen aus Datei"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    self._settings = json.load(f)
            except Exception as e:
                print(f"Error loading settings: {e}")
                self._settings = self._defaults.copy()
        else:
            self._settings = self._defaults.copy()

    def save(self):
        """Speichert Einstellungen in Datei"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self._settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")