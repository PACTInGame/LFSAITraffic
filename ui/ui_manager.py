# ui/ui_manager.py
import time
from typing import Dict, List

import pyinsim
from core.event_bus import EventBus
from core.settings_manager import SettingsManager
from lfs.message_sender import MessageSender



class UIManager:
    """Verwaltet alle UI-Elemente und Menüs"""

    # Buttons:
    # 1-10: HUD-Anzeige
    # 11-12: Forward Collision Warning
    # 13-14: Blind Spot Warning
    # 20-40: Menü-Buttons
    # 41-60: PDC-Anzeige
    # 61: Notifications
    # 62-63: Siren UI

    def __init__(self, event_bus: EventBus, message_sender: MessageSender, settings: SettingsManager):
        self.event_bus = event_bus
        self.message_sender = message_sender
        self.settings = settings
        self.active_elements: Dict[str, bool] = {}
        self.current_menu = None
        self.on_track = False
        self.pdc_data = None
        self.notifications = []
        self.notification_time = time.perf_counter()
        self.collision_warning_level = 0
        self.collision_warning_color = pyinsim.ISB_DARK
        self.hud_enabled = True # TODO get from settings
        self.siren_active = False
        self.strobe_active = False


        # Event-Handler
        self.event_bus.subscribe('collision_warning_changed', self._update_collision_warning_display)
        self.event_bus.subscribe('blind_spot_warning_changed', self._update_blind_spot_display)
        self.event_bus.subscribe('outgauge_data', self._get_hud_data)
        self.event_bus.subscribe('state_data', self._state_change)
        self.event_bus.subscribe("pdc_changed", self._update_pdc)
        self.event_bus.subscribe("notification", self._update_notifications)
        self.event_bus.subscribe("send_lfs_command", self._handle_lfs_command)
        self.event_bus.subscribe("show_siren_ui", self._show_siren_ui)
        self.event_bus.subscribe("button_clicked", self._handle_button_click)
        #self.event_bus.subscribe("decel_debug", self._decel_debug)
        #self.event_bus.subscribe("dist_debug", self._dist_debug)


        self.speed = 0
        self.rpm = 0
        self.gear = 'N'
        self.redline = 0


    def _handle_button_click(self, data):
        btc = data
        button_id = btc.ClickID
        if button_id == 62:
            self.siren_active = not self.siren_active
            self._show_siren_ui({'ui': True})
        elif button_id == 63:
            self.strobe_active = not self.strobe_active
            self._show_siren_ui({'ui': True})


    def _show_siren_ui(self, data):
        ui = data['ui']
        if ui:
            siren_text = "^7Siren" if not self.siren_active else "^4Siren"
            strobe_text = "^7Strobe" if not self.strobe_active else "^4Strobe"
            self.message_sender.create_button(62, self.settings.get("hud_width"), self.settings.get("hud_height") - 5, 6, 5, siren_text, pyinsim.ISB_DARK | pyinsim.ISB_CLICK)
            self.message_sender.create_button(63, self.settings.get("hud_width")+6, self.settings.get("hud_height") - 5, 7, 5, strobe_text,
                                              pyinsim.ISB_DARK | pyinsim.ISB_CLICK)
        else:
            self.message_sender.remove_button(62)

    def _decel_debug(self, data):
        decel = data['deceleration']
        self.message_sender.create_button(100, self.settings.get("hud_width"), self.settings.get("hud_height") - 10,
                                          20, 5,
                                          f"Decel: {decel:.2f} m/s²", pyinsim.ISB_DARK)
    def _dist_debug(self, data):
        decel = data['distance']
        self.message_sender.create_button(101, self.settings.get("hud_width"), self.settings.get("hud_height") - 15,
                                          20, 5,
                                          f"Distance: {decel:.2f} m", pyinsim.ISB_DARK)
    def _handle_lfs_command(self, data):
        command = data['command']
        self.message_sender.send_command(command)

    def _update_notifications(self, data):
        self.notifications.append(data['notification'])
        print(self.notifications)

    def show_notifications(self):
        x = self.settings.get("hud_width")
        y = self.settings.get("hud_height") + 8

        if self.notification_time + 3 < time.perf_counter() and not self.notifications:
            self.message_sender.remove_button(61)
            return
        elif self.notification_time + 3 < time.perf_counter() and self.notifications:
            self.notification_time = time.perf_counter()
            notification_text = self.notifications.pop(0)
            self.message_sender.create_button(61, x, y, 26, 5, notification_text, pyinsim.ISB_DARK)

    def _update_pdc(self, data):
        """Aktualisiert Park Distance Control (PDC) Anzeige"""
        self.pdc_data = data
        if self.pdc_data[0] == -1:
            self.remove_pdc_display()

    def remove_pdc_display(self):
        for i in range(41, 61):
            self.message_sender.remove_button(i)

    def _show_pdc_display(self):
        mode = self.settings.get("park_distance_control_mode", 0)
        if mode > 0:
            top_left = (self.settings.get("hud_width") - 3, self.settings.get("hud_height") - 6)
            bottom_left = (self.settings.get("hud_width") - 3, self.settings.get("hud_height") + 6)
            self.message_sender.create_button(60 , top_left[0] , top_left[1] +6,
                                              3, 2, "^7PDC", pyinsim.ISB_DARK)
            # create buttons for each PDC sensor
            for i, distance in enumerate(self.pdc_data.values()):
                if i < 3:  # Front sensors (0, 1, 2)
                    # Green button (furthest distance)
                    if distance >= 1:
                        self.message_sender.create_button(41 + i, top_left[0] + i, top_left[1],
                                                          1, 2, "^2o", pyinsim.ISB_DARK)
                    else:
                        self.message_sender.remove_button(41 + i)

                    # Yellow button (medium distance)
                    if distance >= 2:
                        self.message_sender.create_button(44 + i, top_left[0] + i, top_left[1] + 2,
                                                          1, 2, "^3o", pyinsim.ISB_DARK)
                    else:
                        self.message_sender.remove_button(44 + i)

                    # Red button (closest distance)
                    if distance >= 3:
                        self.message_sender.create_button(47 + i, top_left[0] + i, top_left[1] + 4,
                                                          1, 2, "^1o", pyinsim.ISB_DARK)
                    else:
                        self.message_sender.remove_button(47 + i)

                else:  # Rear sensors (3, 4, 5)
                    x = i - 3  # Correct offset for rear sensors

                    # Green button (furthest distance) - bottom position for rear
                    if distance >= 1:
                        self.message_sender.create_button(51 + x, bottom_left[0] + x, bottom_left[1],
                                                          1, 2, "^2o", pyinsim.ISB_DARK)
                    else:
                        self.message_sender.remove_button(51 + x)

                    # Yellow button (medium distance)
                    if distance >= 2:
                        self.message_sender.create_button(54 + x, bottom_left[0] + x, bottom_left[1] - 2,
                                                          1, 2, "^3o", pyinsim.ISB_DARK)
                    else:
                        self.message_sender.remove_button(54 + x)

                    # Red button (closest distance) - top position for rear
                    if distance >= 3:
                        self.message_sender.create_button(57 + x, bottom_left[0] + x, bottom_left[1] - 4,
                                                          1, 2, "^1o", pyinsim.ISB_DARK)
                    else:
                        self.message_sender.remove_button(57 + x)
        if mode == 2:
            self.pdc_beeper.beep()



    def _state_change(self, data):
        self.on_track = data['on_track']
        if not self.on_track:
            for i in range(0, 239):
                self.message_sender.remove_button(i)
            self.siren_active = False
            self.strobe_active = False
            self.redline = 0
            self.current_menu = None
            self.message_sender.create_button(1, 0, 180, 25, 5,
                                 "PACT Driving Assist Active.", pyinsim.ISB_DARK)
        else:
            self.message_sender.remove_button(1)

    def _get_hud_data(self, data):
        self.speed = round(data.Speed * 3.6)
        self.rpm = round(data.RPM / 1000, 1)
        self.redline = self.rpm if self.rpm > self.redline else self.redline
        self.gear = "R" if data.Gear == 0 else "N" if data.Gear == 1 else str(data.Gear - 1)

    def update_hud(self):
        """Aktualisiert das Head-Up Display"""
        if self.hud_enabled and self.on_track:
            speed_text = f"{self.speed} km/h" if self.settings.get(
                "unit") == "metric" else f"{round(self.speed * 0.621371)} mph "
            rpm_text = f"{self.rpm} rpm" if self.rpm < self.redline - 1 else f"^1{self.rpm} rpm"
            if self.collision_warning_level >= 2:
                self.collision_warning_color = pyinsim.ISB_LIGHT if self.collision_warning_color == pyinsim.ISB_DARK else pyinsim.ISB_DARK

            if self.collision_warning_level > 0:
                speed_text = "^1- - -"
                rpm_text = "^1- - -"
            if self.collision_warning_level <= 2:
                self.collision_warning_color = pyinsim.ISB_DARK

            self.message_sender.create_button(1, self.settings.get("hud_width"), self.settings.get("hud_height"),
                                                13, 8,
                                                speed_text, self.collision_warning_color)
            self.message_sender.create_button(2, self.settings.get("hud_width") + 13,
                                                  self.settings.get("hud_height"),
                                                  13,
                                                  8, rpm_text, self.collision_warning_color)
            self.message_sender.create_button(3, self.settings.get("hud_width") + 26, self.settings.get("hud_height"),
                                                3, 4,
                                                f"{self.gear}", pyinsim.ISB_DARK)

            if self.pdc_data and self.pdc_data[0] != -1:
                self._show_pdc_display()

            self.show_notifications()



        elif self.on_track:
            self.hide_hud()

    def hide_hud(self):
        """Versteckt das Head-Up Display"""
        for i in range(1, 11):  # HUD button IDs 1-10
            self.message_sender.remove_button(i)

    def _update_collision_warning_display(self, data):
        """Aktualisiert Kollisionswarn-Anzeige"""
        warning_level = data['level']
        if warning_level >= 2 > self.collision_warning_level:
            self.event_bus.emit('play_audio', {'audio_file': 'fcw'})
            self.event_bus.emit('play_audio', {'audio_file': 'fcw'})
            self.event_bus.emit('play_audio', {'audio_file': 'fcw'})

        self.collision_warning_level = warning_level

    def _update_blind_spot_display(self, data):
        """Aktualisiert Toter-Winkel-Anzeige"""
        if data['left']:
            self.message_sender.create_button(13, 20, self.settings.get("hud_height"), 10, 10, "^3!", pyinsim.ISB_DARK)
        else:
            self.message_sender.remove_button(13)

        if data['right']:
            self.message_sender.create_button(14, 180, self.settings.get("hud_height"), 10, 10, "^3!", pyinsim.ISB_DARK)
        else:
            self.message_sender.remove_button(14)
