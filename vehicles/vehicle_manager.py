import time
from typing import Dict, List, Any, Optional

from core.event_bus import EventBus
from vehicles.own_vehicle import OwnVehicle
from vehicles.vehicle import Vehicle


class VehicleManager:
    """Verwaltet alle Fahrzeuge auf der Strecke"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.vehicles: Dict[int, Vehicle] = {}
        self.own_vehicle = OwnVehicle()
        self.players: Dict[int, Any] = {}  # Player info from NPL packets
        self.time_since_last_update = time.perf_counter()
        self.received_cars_count = 0

        # Event-Handler registrieren
        self.event_bus.subscribe('vehicle_data_received', self._handle_vehicle_data)
        self.event_bus.subscribe('player_joined', self._handle_player_joined)
        self.event_bus.subscribe('player_left', self._handle_player_left)
        self.event_bus.subscribe('outgauge_data', self._handle_outgauge_data)

    def _handle_vehicle_data(self, mci_packet):
        """Verarbeitet MCI-Pakete mit Fahrzeugdaten"""
        if self.time_since_last_update + 0.05 < time.perf_counter():
            self.received_cars_count = 0
        self.received_cars_count += len(mci_packet.Info)
        for data in mci_packet.Info:
            player_id = data.PLID

            # Aktualisiere Fahrzeugdaten
            if not self.own_vehicle.data.player_id == player_id:
                # Erstelle Fahrzeug falls nicht vorhanden
                if player_id not in self.vehicles:
                    self.vehicles[player_id] = Vehicle(player_id)
                vehicle = self.vehicles[player_id]

                vehicle.update_position(
                    data.X, data.Y, data.Z,
                    data.Heading, data.Direction,
                    data.Speed / 91.02  # Convert to km/h
                )

                if self.players:
                    vehicle.update_model_and_driver(
                        self.players.get(player_id).get("CName", "Unknown"),
                        self.players.get(player_id).get("PName", "Unknown"),
                        self.players.get(player_id).get("ControlMode", 0)
                    )
            else:
                if self.players:
                    changed = self.own_vehicle.update_model_and_driver(
                        self.players.get(player_id).get("CName", "Unknown"),
                        self.players.get(player_id).get("PName", "Unknown"),
                        self.players.get(player_id).get("ControlMode", 0))
                    if changed:
                        self.event_bus.emit('player_name_changed', {"player_name": self.own_vehicle.data.pname,
                                                                    "control_mode": self.own_vehicle.data.control_mode})

            # if vehicle with own player id is in list, delete it (should not happen, but can happen in first frame)
            if (self.own_vehicle.data.player_id != 0 and
                    self.own_vehicle.data.player_id in self.vehicles and
                    self.vehicles[self.own_vehicle.data.player_id]):
                del self.vehicles[self.own_vehicle.data.player_id]

            if self.own_vehicle.data.player_id == player_id:
                self.own_vehicle.update_position(
                    data.X, data.Y, data.Z,
                    data.Heading, data.Direction,
                    data.Speed / 91.02  # Convert to km/h
                )

        if self.received_cars_count == len(self.players):  # Received all cars for this frame
            if self.own_vehicle.data.player_id != 0:
                for vehicle in self.vehicles.values():
                    vehicle.update_distance_to_player(
                        self.own_vehicle.data.x,
                        self.own_vehicle.data.y,
                        self.own_vehicle.data.z
                    )
                    vehicle.update_angle_to_player(
                        self.own_vehicle.data.x,
                        self.own_vehicle.data.y,
                        self.own_vehicle.data.heading
                    )
            # Emit Event für andere Komponenten

            self.event_bus.emit('vehicles_updated', self.vehicles)

        self.time_since_last_update = time.perf_counter()

    def _get_control_mode(self, flags):
        if len(flags) >= 11 and flags[-11] == 1:
            return 0  # mouse
        elif (len(flags) >= 12 and flags[-12] == 1) or (len(flags) >= 13 and flags[-13] == 1):
            return 1  # keyboard
        else:
            return 2  # wheel

    def _handle_player_joined(self, npl_packet):
        """Verarbeitet neue Spieler"""
        player_info = {
            "PName": npl_packet.PName,
            "CName": npl_packet.CName,
            "ControlMode": self._get_control_mode([int(i) for i in bin(npl_packet.Flags)[2:]])
        }
        self.players[npl_packet.PLID] = player_info
        self.event_bus.emit('player_data_updated', self.players)

    def _handle_player_left(self, pll_packet):
        """Entfernt Spieler"""
        player_id = pll_packet.PLID
        if player_id in self.players:
            del self.players[player_id]
        if player_id in self.vehicles:
            del self.vehicles[player_id]

        self.event_bus.emit('player_data_updated', self.players)

    def _handle_outgauge_data(self, outgauge_packet):
        """Verarbeitet OutGauge-Daten für eigenes Fahrzeug"""
        self.own_vehicle.update_outgauge_data(outgauge_packet)
        self.event_bus.emit('own_vehicle_updated', self.own_vehicle)

    def get_nearby_vehicles(self, max_distance: float = 100.0) -> List[Vehicle]:
        """Gibt nahegelegene Fahrzeuge zurück"""
        return [v for v in self.vehicles.values()
                if v.data.distance_to_player <= max_distance]

    def get_vehicle_by_id(self, player_id: int) -> Optional[Vehicle]:
        """Gibt Fahrzeug anhand der Player-ID zurück"""
        return self.vehicles.get(player_id)
