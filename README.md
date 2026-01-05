# Inertial Kinematics Unlocked (IKU)

Open-source IMU-driven kinematics + visualization toolkit.

> **Status:** early / research-grade. Contributions welcome.

## What’s in this repo
- compiled version IKU.exe for windows
- `iku/` — Python source
-        - environment.yaml
-        — logo

Instructions - Download the Zip File 'IKU' and extract.  Click IKU.exe and the application should load.  Follow the on-screen instructions.  


This project has been designed around streaming data in from xsens awinda IMUs.  If you can stream data from your IMUs then I'm sure you can build a module to import it into the application.  

## Xsens Awinda / MTw support - This project requires MT Manager and python bindings from the MT SDK to run.
IF NO CONNECTION IS MADE BETWEEN A MASTER AND SENSORS IN MT MANAGER, A FAKE IMU STREAM OF SOME RANDOM MOTIONS WILL BE USED TO ALLOW DEMO OF THE SOFTWARE>

Please install MT manager (2025) using the MT Software Suite Download.https://www.xsens.com/support/software-documentation
This project imports `xsensdeviceapi`, Which comes from the MT SDK (typically found C:\Program Files\Xsens\MT Software Suite 2025.0\MT SDK\Python\x64) on windows.  You will need to use a python version for which there is an available .whl
That package is **not** installable via `pip` — you must install the SDK and ensure the Python
bindings are on your `PYTHONPATH` (or installed into your venv).

Sensor Mounting. All with charging port facing down in N pose
Sensor mounting orientations can be edited in the mounting.py file.  Standard mountings as currently coded (on velcro straps) are as follows:
Pelvis: Posterior
Torso: Standard Shirt mounting
Head: Posterior
Upper Arms: Outer
Forearms - Dorsal aspect of wrist.
Hands - Standard gloves (back of hand)
Thighs - Outer
Shins - Outer
Feet - Dorsum, charging port towards toes. 

I'm sure I've left stuff out, hopefully you can work it out, or ask.  


## License

MIT (see `LICENSE`).

## Citation

If you use IKU in academic work, please cite the repository and (optionally) include a short methods
description of your calibration + kinematics pipeline.
