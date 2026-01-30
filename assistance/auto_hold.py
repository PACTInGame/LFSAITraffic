from typing import Dict, Any
import pyautogui
from assistance.base_system import AssistanceSystem
from core.event_bus import EventBus
from core.settings_manager import SettingsManager
from vehicles.own_vehicle import OwnVehicle
from vehicles.vehicle import Vehicle


class AutoHold(AssistanceSystem):
    """Automatic Parking Brake"""

    def __init__(self, event_bus: EventBus, settings: SettingsManager):
        super().__init__("auto_hold", event_bus, settings)
        self.current_warning_level = 0
        self.own_rectangle = None

    def process(self, own_vehicle: OwnVehicle, vehicles: Dict[int, Vehicle]) -> Dict[str, Any]:
        """Verarbeitet die Auto-Hold-Logik"""
        if not self.is_enabled():
            return {'auto_hold_active': False}
        auto_hold = False
        if own_vehicle.data.speed < 0.05 and own_vehicle.brake > 0.05:
            auto_hold = True
            if not own_vehicle.handbrake_light:
                user_handbrake_key = self.settings.get('user_handbrake_key')
                # Press the handbrake key to activate auto-hold using direct input, right here
                pyautogui.keyDown(user_handbrake_key)
                pyautogui.keyUp(user_handbrake_key)
                self.event_bus.emit("notification", {'notification': 'Auto Hold'})

        return {
            'auto_hold_active': auto_hold
        }
