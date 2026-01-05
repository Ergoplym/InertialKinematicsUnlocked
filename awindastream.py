"""
awindastream.py

Awinda MTw live quaternion streamer for Xsens Device API (MT Software Suite 2022.2).

Provides:
- AwindaStream(mapping_csv="mapping.csv")
    - start()
    - stop()
    - get_latest_quats()  -> {segment_name: np.array([w, x, y, z])}
- _demo_cli() when run as a script:
    python awindastream.py

Requires:
- xsensdeviceapi 2022.2 wheel for Python 3.10
- numpy (<= 1.23.x recommended)
- pandas
"""

import threading
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import Xsens Device API
# ---------------------------------------------------------------------------

try:
    import xsensdeviceapi as xda
    if not hasattr(xda, "XsControl_construct"):
        from xsensdeviceapi import xsensdeviceapi_py310_64 as xda  # type: ignore
except ImportError:
    from xsensdeviceapi import xsensdeviceapi_py310_64 as xda  # type: ignore


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------

class AwindaCallback(xda.XsCallback):
    """
    Custom callback handler to store latest quaternions for each device.
    """

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        # device_id (int) -> np.array([w,x,y,z])
        self._latest_quats: Dict[int, np.ndarray] = {}

    def onLiveDataAvailable(self, dev, packet):
        """
        Called by the Xsens API when a new data packet arrives.
        Handles both XsQuaternion-style objects and numpy arrays.
        """
        try:
            if not packet.containsOrientation():
                return

            quat = packet.orientationQuaternion()
            dev_id_obj = dev.deviceId()

            # Resolve device ID to int
            try:
                dev_id = int(dev_id_obj.toInt())
            except Exception:
                dev_id = int(str(dev_id_obj), 16)

            # Handle both possible types: XsQuaternion or numpy.ndarray
            if hasattr(quat, "w") and callable(quat.w):
                # Xsens quaternion object style
                w = float(quat.w())
                x = float(quat.x())
                y = float(quat.y())
                z = float(quat.z())
                q = np.array([w, x, y, z], dtype=float)
            else:
                # Assume it's already an array-like [w, x, y, z] or similar
                arr = np.asarray(quat, dtype=float).ravel()
                if arr.size != 4:
                    print(
                        f"[AwindaCallback] Unexpected quaternion shape from "
                        f"orientationQuaternion(): {arr.shape}"
                    )
                    return
                q = arr

            with self._lock:
                self._latest_quats[dev_id] = q

        except Exception as e:
            print(f"[AwindaCallback] Error in onLiveDataAvailable: {e}")

    def get_latest(self) -> Dict[int, np.ndarray]:
        """
        Thread-safe access to latest quaternions.

        Returns:
            dict: {device_id : np.array([w,x,y,z])}
        """
        with self._lock:
            return dict(self._latest_quats)


# ---------------------------------------------------------------------------
# AwindaStream class
# ---------------------------------------------------------------------------

class AwindaStream:
    """
    High-level wrapper around Awinda station.

    mapping_csv should have:
        IMU_ID,Segment_Name
    where IMU_ID is hex string (e.g. '00B4A89F'), Segment_Name is the label
    used in your biomechanics model (e.g. 'pelvis_imu', 'torso_imu', etc.).
    """

    def __init__(
        self,
        mapping_csv: str,
        radio_channel: int = 11,
        timeout: float = 2.0,
        sensor_connect_wait: float = 2.0,
    ):
        """
        :param mapping_csv: CSV mapping IMU_ID(hex) -> Segment_Name
        :param radio_channel: Awinda radio channel to enforce (11 for your setup)
        :param timeout: time to wait for first data packet after starting measurement
        :param sensor_connect_wait: time to wait after radio is confirmed for sensors to connect
        """
        self.mapping_csv = mapping_csv
        self.radio_channel = radio_channel
        self.timeout = timeout
        self.sensor_connect_wait = sensor_connect_wait

        # dev_id (int) -> segment_name
        self._id_to_segment = self._load_mapping(mapping_csv)

        self._callback = AwindaCallback()
        self._control: Optional[xda.XsControl] = None
        self._master: Optional[xda.XsDevice] = None  # type: ignore
        self._started = False

    # ------------------ PUBLIC API ------------------

    def start(self):
        """Connect to Awinda master and start streaming."""
        if self._started:
            return
        print("[AwindaStream] Setting up XDA...")
        self._setup_xda()
        print("[AwindaStream] Configuring master (radio, etc.)...")
        self._configure()
        print("[AwindaStream] Starting measurement...")
        self._start_measurement()
        self._started = True
        print("[AwindaStream] Awinda streaming started.")

    def stop(self):
        """Stop streaming + close port."""
        print("[AwindaStream] Stopping stream and closing port...")
        try:
            if self._master is not None and self._master.isMeasuring():
                self._master.gotoConfig()
        except Exception as e:
            print(f"[AwindaStream] Warning during gotoConfig() in stop: {e}")

        try:
            if self._master is not None:
                self._master.removeCallbackHandler(self._callback)
        except Exception:
            pass

        try:
            if self._control is not None:
                self._control.close()
        except Exception:
            pass

        self._master = None
        self._control = None
        self._started = False
        print("[AwindaStream] Stopped.")

    def get_latest_quats(self) -> Dict[str, np.ndarray]:
        """
        Returns:
            dict: {segment_name : np.array([w,x,y,z])}
        """
        raw = self._callback.get_latest()
        segs: Dict[str, np.ndarray] = {}
        for dev_id, q in raw.items():
            seg = self._id_to_segment.get(dev_id)
            if seg is not None:
                segs[seg] = q
            else:
                # If unmapped, expose as UNMAPPED_<hex_id>
                segs[f"UNMAPPED_{dev_id:X}"] = q
        return segs

    # ------------------ INTERNALS ------------------

    def _load_mapping(self, path: str) -> Dict[int, str]:
        """Load mapping.csv: IMU_ID(hex) -> Segment_Name."""
        df = pd.read_csv(path)
        if "IMU_ID" not in df.columns or "Segment_Name" not in df.columns:
            print(
                "[AwindaStream] WARNING: mapping file loaded but no valid rows "
                "(check column names: 'IMU_ID', 'Segment_Name')."
            )
            return {}

        mapping: Dict[int, str] = {}
        for _, row in df.iterrows():
            imu = str(row["IMU_ID"]).strip()
            seg = str(row["Segment_Name"]).strip()
            if not imu or not seg:
                continue
            try:
                dev_id = int(imu, 16)
                mapping[dev_id] = seg
            except ValueError:
                print(f"[AwindaStream] Warning: invalid IMU_ID hex '{imu}' in mapping.")
        if not mapping:
            print(
                "[AwindaStream] WARNING: mapping CSV produced no valid rows "
                "(check IMU_ID hex strings)."
            )
        else:
            print(f"[AwindaStream] Loaded {len(mapping)} IMU mappings from {path}")
        return mapping

    def _debug_dump_master_state(self, where: str):
        """
        Dump as much state info as possible about the master and its children.
        """
        if not self._master:
            print(f"[AwindaStream] DEBUG ({where}): no master device.")
            return

        d = self._master
        print(f"[AwindaStream] DEBUG ({where}): --- master state dump ---")

        for attr in ["deviceId", "deviceState", "isMeasuring", "isRadioEnabled", "radioChannel"]:
            try:
                print(f"  {attr}: {getattr(d, attr)()}")
            except Exception:
                pass

        # children
        try:
            count = d.childCount()
            print(f"  childCount: {count}")
            for i, ch in enumerate(d.children()):
                try:
                    print(
                        f"    child[{i}]: id={ch.deviceId()}, "
                        f"state={ch.deviceState()}, "
                        f"connectivityState={ch.connectivityState()}, "
                        f"isMeasuring={ch.isMeasuring()}"
                    )
                except Exception as e:
                    print(f"    child[{i}]: error: {e}")
        except Exception as e:
            print(f"  error enumerating children: {e}")

        try:
            print(f"  lastResult: {d.lastResult()} {d.lastResultText()}")
        except Exception:
            pass

        try:
            print(
                f"  control lastResult: {self._control.lastResult()} {self._control.lastResultText()}"
            )
        except Exception:
            pass

        print(f"[AwindaStream] DEBUG ({where}): --- end dump ---")

    def _setup_xda(self):
        """Find Awinda station, open port, attach callback."""
        self._control = xda.XsControl_construct()
        if self._control is None:
            raise RuntimeError("Failed to construct XsControl.")

        ports = xda.XsScanner_scanPorts()

        try:
            n_ports = ports.size()
        except Exception:
            n_ports = len(ports)

        print(f"[AwindaStream] Found {n_ports} Xsens port(s).")
        if n_ports == 0:
            raise RuntimeError("No Xsens devices detected.")

        master_port = None
        for i in range(n_ports):
            p = ports[i]
            did = p.deviceId()
            is_master = (
                (hasattr(did, "isWirelessMaster") and did.isWirelessMaster()) or
                (hasattr(did, "isAwindaStation") and did.isAwindaStation()) or
                (hasattr(did, "isAwindaDongle") and did.isAwindaDongle())
            )
            print(f"[AwindaStream] Port {i}: {p.portName()}, master={is_master}")
            if is_master and master_port is None:
                master_port = p

        if master_port is None:
            print("[AwindaStream] No explicit master found; using first port.")
            master_port = ports[0]

        port_name = master_port.portName()
        baud = master_port.baudrate()

        print(f"[AwindaStream] Opening port {port_name} at baud {baud}...")
        if not self._control.openPort(port_name, baud, 0, True):
            print("[AwindaStream] ERROR: Failed to open port", port_name)
            raise RuntimeError(f"Failed to open port {port_name}")

        self._master = self._control.device(master_port.deviceId())
        if self._master is None:
            raise RuntimeError("Failed to get master device")

        print(f"[AwindaStream] Master device: {self._master.deviceId()}")
        self._master.addCallbackHandler(self._callback)

        print("[AwindaStream] Putting master into config mode...")
        if not self._master.gotoConfig():
            self._debug_dump_master_state("after gotoConfig (FAILED)")
            raise RuntimeError(
                "Failed to enter config mode. "
                f"Device lastResult: {self._master.lastResult()} {self._master.lastResultText()}"
            )

        self._debug_dump_master_state("after gotoConfig")

    # ------------------------ _configure ------------------------

    def _configure(self):
        """
        Configure the Awinda master.

        - Log access control mode (actual behavior is managed via MT Manager).
        - Force radio to self.radio_channel (11 in your case).
        - Then call makeOperational(), wait for sensors, and log mapping coverage.
        """
        if self._master is None:
            raise RuntimeError("Master device is not initialised.")

        # ---- ACCESS CONTROL: just log it (we assume MT Manager has configured it) ----
        try:
            acm = self._master.accessControlMode()
            print(f"[AwindaStream] Current access control mode (raw int): {acm}")
        except Exception as e:
            print(f"[AwindaStream] Could not query accessControlMode(): {e}")

        # ---- FORCE RADIO TO CHANNEL 11 ----
        try:
            desired = self.radio_channel  # 11
            radio_on = False
            current_channel = -1

            try:
                radio_on = bool(self._master.isRadioEnabled())
            except Exception:
                radio_on = False

            try:
                current_channel = int(self._master.radioChannel())
            except Exception:
                current_channel = -1

            if (not radio_on) or (current_channel != desired):
                print(
                    f"[AwindaStream] Setting radio to channel {desired} "
                    f"(was {current_channel if current_channel != -1 else 'UNKNOWN'}, "
                    f"radio_on={radio_on})."
                )
                if not hasattr(self._master, "enableRadio"):
                    raise RuntimeError("Master device has no enableRadio() method.")

                ok = self._master.enableRadio(desired)
                print(f"[AwindaStream] Radio enable result: {ok}")

                if not ok:
                    msg = ""
                    try:
                        msg = self._master.lastResultText()
                    except Exception:
                        pass
                    self._debug_dump_master_state("after enableRadio (FAILED)")
                    raise RuntimeError(
                        f"Failed to enable radio: {msg}"
                    )
            else:
                print(
                    f"[AwindaStream] Radio already enabled on desired channel {desired}, "
                    "no change needed."
                )

        except Exception as e:
            raise RuntimeError(f"[AwindaStream] Radio configuration failed: {e}")

        # ---- OPTIONAL: makeOperational ----
        if hasattr(self._master, "makeOperational"):
            try:
                print("[AwindaStream] Calling makeOperational()...")
                op_ok = self._master.makeOperational()
                print(f"[AwindaStream] makeOperational() result: {op_ok}")
            except Exception as e:
                print(f"[AwindaStream] Warning: makeOperational() exception: {e}")

        self._debug_dump_master_state("before sensor connect wait")

        # ---- WAIT FOR SENSORS ----
        wait_s = self.sensor_connect_wait
        print(
            f"[AwindaStream] Waiting up to {wait_s:.1f} s for MTw sensors to connect..."
        )

        start = time.monotonic()
        last_count = -1

        while True:
            elapsed = time.monotonic() - start
            if elapsed > wait_s:
                break

            try:
                count = self._master.childCount()
                children = self._master.children()
            except Exception:
                count = 0
                children = []

            if count != last_count:
                print(
                    f"[AwindaStream] {count} sensor(s) connected after "
                    f"{elapsed:.1f} s."
                )
                if count > 0:
                    print("[AwindaStream] Connected sensor IDs:")
                    for ch in children:
                        try:
                            cid = ch.deviceId()
                            try:
                                cid_int = int(cid.toInt())
                                cid_hex = f"{cid_int:08X}"
                            except Exception:
                                cid_str = str(cid)
                                cid_int = int(cid_str, 16)
                                cid_hex = f"{cid_int:08X}"
                            print(f"  - {cid} (int={cid_int}, hex={cid_hex})")
                        except Exception as e:
                            print(f"  - error: {e}")
                last_count = count

            time.sleep(0.5)

        # Final dump
        self._debug_dump_master_state("after sensor connect wait")

        try:
            final_count = self._master.childCount()
        except Exception:
            final_count = 0

        if final_count == 0:
            print(
                "[AwindaStream] WARNING: No MTw sensors connected. "
                "If sensors are on and flashing, you may need to configure or "
                "repair the Awinda wireless network in MT Manager (add sensors "
                "to station and set channel 11)."
            )
        else:
            children = self._master.children()
            mapped = 0
            for ch in children:
                try:
                    cid_obj = ch.deviceId()
                    try:
                        cid_int = int(cid_obj.toInt())
                    except Exception:
                        cid_int = int(str(cid_obj), 16)

                    seg = self._id_to_segment.get(cid_int)
                    if seg:
                        mapped += 1
                        print(f"[AwindaStream] Mapping: {cid_obj} -> {seg}")
                    else:
                        print(
                            f"[AwindaStream] Mapping WARNING: {cid_obj} "
                            "not found in mapping.csv"
                        )
                except Exception as e:
                    print(f"[AwindaStream] Mapping WARNING: {e}")

            print(
                f"[AwindaStream] Sensor mapping coverage: "
                f"{mapped}/{final_count} sensors mapped."
            )

    # ------------------------ START MEASUREMENT ------------------------

    def _start_measurement(self):
        """Enter measurement mode and wait for orientation packets."""
        if self._master is None:
            raise RuntimeError("Master not initialised.")

        print("[AwindaStream] Switching master to measurement mode...")
        ok = self._master.gotoMeasurement()

        if not ok:
            self._debug_dump_master_state("after gotoMeasurement (FAILED)")
            m = self._master
            raise RuntimeError(
                "Could not start measurement mode.\n"
                f"Device lastResultText: {m.lastResultText()}"
            )

        print("[AwindaStream] Master is in measurement mode, waiting for data...")
        self._debug_dump_master_state("after gotoMeasurement (SUCCESS)")

        start = time.monotonic()
        while time.monotonic() - start < self.timeout:
            data = self._callback.get_latest()
            if data:
                print("[AwindaStream] First data packet received.")
                return
            time.sleep(0.1)

        raise RuntimeError("Measurement started, but no data received within timeout.")


# ---------------------------------------------------------------------------
# Demo CLI
# ---------------------------------------------------------------------------

def _demo_cli(mapping_csv: str = "mapping.csv"):
    print("Starting Awinda demo test...")
    stream = AwindaStream(
        mapping_csv=mapping_csv,
        radio_channel=11,      # staying on your current channel
        timeout=2.0,
        sensor_connect_wait=2.0,
    )
    try:
        stream.start()
        print("Awinda stream started. Press CTRL+C to stop.")
        while True:
            quats = stream.get_latest_quats()
            if quats:
                # Print one example segment to show it's alive
                seg_name, q = next(iter(quats.items()))
                print(f"{seg_name}: {q}")
            else:
                print("No data yet (check MTw sensors are undocked and on).")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[AwindaStream] Keyboard interrupt, stopping...")
    except Exception as e:
        print(f"[AwindaStream] ERROR: {e}")
    finally:
        stream.stop()


if __name__ == "__main__":
    _demo_cli("mapping.csv")
