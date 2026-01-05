# Inertial Kinematics Unlocked (IKU)

Open-source IMU-driven kinematics + visualization toolkit.

> **Status:** early / research-grade. Contributions welcome.

## What’s in this repo

- `iku/` — Python source (PyQt GUI + kinematics / calibration modules)
-        — logo and other media

## Xsens Awinda / MTw support

This project imports `xsensdeviceapi`, which comes from the **Xsens Device API / Movella SDK**.
That package is **not** installable via `pip` — you must install the SDK and ensure the Python
bindings are on your `PYTHONPATH` (or installed into your venv).

Typical approaches:
- Install the Xsens/Movella SDK, then add its `python` bindings folder to `PYTHONPATH`
- Or copy/pack the `xsensdeviceapi` module into your environment (per the SDK license)


## License

MIT (see `LICENSE`).

## Citation

If you use IKU in academic work, please cite the repository and (optionally) include a short methods
description of your calibration + kinematics pipeline.
