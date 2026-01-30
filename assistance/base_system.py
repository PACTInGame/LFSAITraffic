# assistance/base_system.py
from abc import ABC, abstractmethod
from typing import Dict, Any

from core.event_bus import EventBus
from core.settings_manager import SettingsManager
from vehicles.own_vehicle import OwnVehicle
from vehicles.vehicle import Vehicle


class AssistanceSystem(ABC):
    """Basis-Klasse für alle Fahrerassistenzsysteme"""

    def __init__(self, name: str, event_bus: EventBus, settings: SettingsManager):
        self.name = name
        self.event_bus = event_bus
        self.settings = settings
        self.enabled = True
        self.last_update = 0

        # Standard Events abonnieren
        self.event_bus.subscribe('vehicles_updated', self.on_vehicles_updated)
        self.event_bus.subscribe('own_vehicle_updated', self.on_own_vehicle_updated)

    @abstractmethod
    def process(self, own_vehicle: OwnVehicle, vehicles: Dict[int, Vehicle]) -> Dict[str, Any]:
        """Hauptverarbeitungslogik des Assistenzsystems"""
        pass

    def on_vehicles_updated(self, vehicles: Dict[int, Vehicle]):
        """Called when vehicle data is updated"""
        if self.enabled:
            # Subklassen können dies überschreiben
            pass

    def on_own_vehicle_updated(self, own_vehicle: OwnVehicle):
        """Called when own vehicle data is updated"""
        if self.enabled:
            # Subklassen können dies überschreiben
            pass

    def is_enabled(self) -> bool:
        """Prüft ob System aktiviert ist"""
        return self.enabled and self.settings.get(f'{self.name.lower()}', True)
