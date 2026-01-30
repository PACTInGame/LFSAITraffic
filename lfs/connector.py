import pyinsim
from typing import Dict, Any, Callable
import time

from AI_Control import AICarController
from core.event_bus import EventBus
from core.settings_manager import SettingsManager
from lfs.lfs_state import StateHandler


class LFSConnector:
    """Verwaltet die Verbindung zu Live for Speed"""

    def __init__(self, event_bus: EventBus, settings: SettingsManager):
        self.debug = False
        self.event_bus = event_bus
        self.settings = settings
        self.insim = None
        self.outgauge = None
        self.outsim = None
        self.is_connected = False
        self._packet_handlers: Dict[int, Callable] = {}
        self._setup_handlers()
        self.event_bus.subscribe("send_light_command", self.send_light_command)
        self.event_bus.subscribe("request_axm_update", self._request_axm_update)
        self.event_bus.subscribe("siren_state_changed", self._siren_state_changed)
        self.AIController = None


    def _siren_state_changed(self, data):
        siren = data['siren_active']
        siren_config =  (pyinsim.LCS_SET_SIREN, pyinsim.LCS_Mask_Siren)


        set_flag, mask_flag = siren_config
        UVal = set_flag | (mask_flag if siren else 0)

        self.insim.send(pyinsim.ISP_SMALL, SubT=pyinsim.SMALL_LCS, UVal=UVal)

    def _request_axm_update(self, data):
        """Fordert ein AXM-Layout-Update an"""
        self.insim.send(pyinsim.ISP_TINY, ReqI=255, SubT=pyinsim.TINY_AXM)

    def _setup_handlers(self):
        """Registriert Standard-Packet-Handler und startet listener"""

        self.state_handler = StateHandler(self)

        self._packet_handlers = {
            pyinsim.ISP_NPL: self._handle_new_player,
            pyinsim.ISP_PLL: self._handle_player_left,
            pyinsim.ISP_STA: self._handle_state,
            pyinsim.ISP_BTC: self._handle_button_click,
            pyinsim.ISP_MSO: self._handle_message,
            pyinsim.ISP_MCI: self._handle_mci,
            pyinsim.ISP_AXM: self._handle_layout,
        }

    def connect(self):
        """Stellt Verbindung zu LFS her"""

        try:
            interval = self.settings.get('assistance_refresh_rate')
            self.insim = pyinsim.insim(
                b'127.0.0.1', 29999,
                Admin=b'',
                Prefix=b"$",
                Flags=pyinsim.ISF_MCI | pyinsim.ISF_AXM_LOAD | pyinsim.ISF_AXM_EDIT | pyinsim.ISF_LOCAL,
                Interval=interval
            )

            # Registriere alle Handler
            for packet_type, handler in self._packet_handlers.items():
                self.insim.bind(packet_type, handler)

            self.start_outgauge()
            self.start_outsim()

            self.is_connected = True
            self.event_bus.emit('lfs_connected')

            self.insim.send(pyinsim.ISP_TINY, ReqI=255, SubT=pyinsim.TINY_SST)
            self.insim.send(pyinsim.ISP_TINY, ReqI=255, SubT=pyinsim.TINY_AXM)
            self.AIController = AICarController(self.insim)
            self.event_bus.emit('AI_Controller_initialized', self.AIController)


        except Exception as e:
            print(f"Failed to connect to LFS: {e}")
            self.is_connected = False

    def start_outgauge(self):
        """Startet OutGauge-Verbindung"""
        try:
            self.outgauge = pyinsim.outgauge('127.0.0.1', 30000, self._outgauge_handler, 30.0)
        except Exception as e:
            print(f"Failed to connect OutGauge: {e}")

    def start_outsim(self):
        """Startet OutSim-Verbindung"""
        try:
            self.outsim = pyinsim.outsim('127.0.0.1', 29998, self._outsim_handler, 30.0)
        except Exception as e:
            print(f"Failed to connect OutSim: {e}")

    def _handle_mci(self, insim, mci):
        """Verarbeitet MCI (Multi Car Info) Pakete"""
        self.event_bus.emit('vehicle_data_received', mci)

    def _handle_new_player(self, insim, npl):
        """Verarbeitet neue Spieler"""

        self.event_bus.emit('player_joined', npl)

    def _handle_player_left(self, insim, pll):
        """Verarbeitet Spieler die verlassen"""
        self.event_bus.emit('player_left', pll)

    def _handle_state(self, insim, sta):
        """Verarbeitet Spielstatus"""
        self.event_bus.emit('game_state_changed', sta)

    def _handle_button_click(self, insim, btc):
        """Verarbeitet Button-Klicks"""
        self.event_bus.emit('button_clicked', btc)

    def _handle_message(self, insim, mso):
        """Verarbeitet Chat-Nachrichten"""
        self.event_bus.emit('message_received', mso)

    def _handle_layout(self, insim, axm):
        """Verarbeitet Layout-Pakete"""
        self.event_bus.emit('layout_received', axm)
        print("DEBUG: AXM layout received with", len(axm.Info), "objects.")

    def _outgauge_handler(self, outgauge, packet):
        """Handler für OutGauge-Pakete (hochfrequent)"""
        self.event_bus.emit('outgauge_data', packet)

    def _outsim_handler(self, outsim, packet):
        """Handler für OutSim-Pakete"""
        self.event_bus.emit('outsim_data', packet)

    def send_command_to_lfs(self, command: str):
        """Sendet einen Befehl an LFS"""
        command = command.encode()
        self.insim.send(pyinsim.ISP_MST,
                        Msg=command)

    def send_light_command(self, data):
        """Schaltet ein Licht ein oder aus

        Args:
            light: 0=Standlicht, 1=Abblendlicht, 2=Fernlicht, 3=Nebelscheinwerfer,
                   4=Nebelschlussleuchte, 5=Extra, 6=Blinker links, 7=Blinker rechts, 8=Warnblinkanlage
            on: True zum Einschalten, False zum Ausschalten
        """
        light = data['light']
        on = data['on']
        # Mapping: light_id -> (SET_flag, MASK_flag)
        light_config = {
            0: (pyinsim.LCL_SET_LIGHTS, pyinsim.LCL_Mask_SideLight),      # Standlicht
            1: (pyinsim.LCL_SET_LIGHTS, pyinsim.LCL_Mask_LowBeam),        # Abblendlicht
            2: (pyinsim.LCL_SET_LIGHTS, pyinsim.LCL_Mask_HighBeam),       # Fernlicht
            3: (pyinsim.LCL_SET_FOG_FRONT, pyinsim.LCL_Mask_FogFront),    # Nebelscheinwerfer
            4: (pyinsim.LCL_SET_FOG_REAR, pyinsim.LCL_Mask_FogRear),      # Nebelschlussleuchte
            5: (pyinsim.LCL_SET_EXTRA, pyinsim.LCL_Mask_Extra),           # Extra
            6: (pyinsim.LCL_SET_SIGNALS, pyinsim.LCL_Mask_Left),          # Blinker links
            7: (pyinsim.LCL_SET_SIGNALS, pyinsim.LCL_Mask_Right),         # Blinker rechts
            8: (pyinsim.LCL_SET_SIGNALS, pyinsim.LCL_Mask_Signals),       # Warnblinkanlage
        }

        if light not in light_config:
            print("DEBUG: CAUTION: Invalid light ID")
            return

        set_flag, mask_flag = light_config[light]
        UVal = set_flag | (mask_flag if on else 0)

        self.insim.send(pyinsim.ISP_SMALL, SubT=pyinsim.SMALL_LCL, UVal=UVal)

    def send_button(self, click_id: int, style: int, t: int, l: int, w: int, h: int, text: str, inst: int = 0):
        """Sendet einen Button an LFS (T < 170 überlappt UI von LFS)"""
        # print(f"ClickID: {click_id}, Style: {style}, Position: ({t}, {l}), Size: ({w}, {h}), Text: '{text}'")
        try:
            text = text.encode("latin-1")
        except UnicodeEncodeError:
            text = text.encode()
        if self.insim and self.is_connected:
            self.insim.send(
                pyinsim.ISP_BTN,
                ReqI=255,
                ClickID=click_id,
                BStyle=style | 3,
                Inst=inst,
                T=t, L=l, W=w, H=h,
                Text=text,
                TypeIn=0
            )

    def delete_button(self, click_id: int):
        """Löscht einen Button in LFS"""
        if self.insim and self.is_connected:
            self.insim.send(pyinsim.ISP_BFN, ReqI=255, ClickID=click_id)
