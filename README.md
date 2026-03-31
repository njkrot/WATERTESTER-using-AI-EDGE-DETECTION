# Water Filter Tester - Pi 5

my water quality testing project. uses a raspi 5 with an IMX500 camera to read 16-pad test strips and score the water. has pumps and solenoids to fill a test container, read the strip, dump the test water, then filter the actual water. theres a 3.5" touchscreen for the ui

## quick start on the Pi

1. clone or copy this repo to e.g. `/home/pi/filterprogram`
2. install system packages (see below)
3. `chmod +x run.sh` then `./run.sh`  
   - for a normal window (hdmi monitor / debugging): `FILTER_WINDOWED=1 ./run.sh`
4. strip reading without YOLO: `python3 GRID_READER.py` (fixed camera crop)

## how it works

1. press START on the touchscreen
2. pump fills test container
3. strip soaks ~5 sec
4. solenoid opens, dumps test water out
5. camera reads the strip - keeps taking frames until it has enough good ones, throws out bad reads
6. shows score + warnings (lead/mercury weigh 3x more than like fluoride)
7. main pump runs 100 sec through the filter path (plumbing, no diverter gpio)
8. press DISPENSE for clean water

## scoring

0-100 weighted scale, bad stuff counts more

- lead, mercury = 3x weight
- copper, iron, manganese = 2x
- free chlorine, total chlorine, h2s, fluoride = 1.5x
- everything else = 1x

if the waters bad it still filters it just warns you whats wrong

## GPIO map (BCM numbers, matches gpiozero)

outputs (relays):

| BCM | physical | role in code |
|-----|----------|----------------|
| 18 | pin 12 | filter pump (`FILTER_PUMP_PIN`) - DISPENSE step |
| 27 | pin 13 | main 24v pump (`MAIN_PUMP_PIN`) - fill + filter timer |
| 22 | pin 15 | dump solenoid (`DUMP_VALVE_PIN`) |
| 23 | pin 16 | aux solenoid (`AUX_VALVE_PIN`) - off unless `OPEN_AUX_ON_DRAIN` |
| 6 | pin 31 | UV (`UV_PIN`) - optional, see `USE_UV_WHILE_FILTERING` in `pump_control.py` |

inputs:

| BCM | physical | role |
|-----|----------|------|
| 12 | pin 32 | tank level (read in `get_status()` / `read_sensors()`) |
| 13 | pin 33 | flow sensor |

other:

| BCM | role |
|-----|------|
| 17 | cartridge detect switch (`PROGRAM.py` / `GRID_READER.py`) |
| 19 | status LED (`STATUS_LED_PIN`) - **not** 27 (27 is the main pump relay) |

keep I2C (2,3), UART (14,15), and ID eeprom pins (0,1) free per your wiring notes.

### Waveshare 3.5" LCD (A) + GPIO 18 clash

a lot of Waveshare SPI configs use **GPIO 18** for backlight PWM. that fights **relay IN1** if your filter pump is on 18. fixes people actually use:

- move that relay input to another free BCM and change `FILTER_PUMP_PIN` in `pump_control.py`, or
- follow Waveshares wiki to use a backlight setting that doesnt steal 18 (depends on their `LCD-show` / overlay version)

the app assumes the framebuffer is already the 3.5 screen (their installer usually sets that). this isnt a second hdmi monitor - you install their driver once.

### IMX500 camera not showing up

- use **Raspberry Pi OS (64-bit) Bookworm** on Pi 5 for the least pain; newer trixie/sid can lag on imx500 firmware.
- `sudo raspi-config` → enable camera / latest camera stack
- `sudo apt update && sudo apt full-upgrade` then reboot
- test: `rpicam-hello` or `libcamera-hello` (package name varies slightly by image)
- IMX500 needs the imx500 packages + correct dtoverlay per [Raspberry Pi IMX500 docs](https://www.raspberrypi.com/documentation/accessories/camera.html) - if `rpicam-hello` works but this app doesnt, check `MODEL_PATH` / `LABELS_PATH` in `PROGRAM.py` point at your exported `.rpk` folder

**GRID mode** (`GRID_READER.py` / touch UI crop path) only needs a working picamera2 sensor - no IMX500 AI blob required.

## hardware summary

- raspberry pi 5
- camera: IMX500 for `PROGRAM.py`, or any picamera2 camera for grid mode
- Waveshare 3.5" (A) or any display tk can use
- relays: main pump, filter/dispense pump, dump + optional aux solenoid
- optional: UV on GPIO 6, tank level 12, flow 13
- cartridge switch gpio 17, status led gpio 19
- (no software-controlled diverter - mechanical plumbing)

## deps

- python 3.9+
- opencv, numpy
- picamera2, gpiozero
- tkinter
- ultralytics (only for training / imx export on the pi)

### ultralytics pip failures on Pi

dont fight the system python if wheels fail:

```bash
sudo apt install -y python3-venv python3-pip build-essential
cd /home/pi/filterprogram
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics numpy
```

for export only you can use the venv then: `python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='imx', imgsz=640)"`

## files

- `display.py` - **all UI**: 3.5" touch app (`python3 display.py` / `./run.sh`), character LCD, tk lcd sim, console; `Display` class for `PROGRAM.py`
- `run.sh` - cd to folder and run `display.py` (Pi)
- `pump_control.py` - pump, valves, optional UV + sensor reads
- `PROGRAM.py` - YOLO + color analysis pipeline
- `GRID_READER.py` - simpler reader, no AI, just fixed crop
- `StripEdgeRefinement.py` - edge cleanup
- `prepare_water_dataset.py` - turns strip photos into yolo dataset
- `train_model.py` - trains the model
- `test_model.py` - tests the model on images
- `test_offline.py` - offline tests, no hardware needed

## training

```
pip install pillow-heif ultralytics pyyaml
python prepare_water_dataset.py
python train_model.py
```

auto finds strips in your photos, makes annotations, trains yolo11s at 640px. gets like 0.995 mAP on our strips

test it:
```
python test_model.py
python test_model.py path/to/img.jpg
```

## putting it on the pi

1. copy stuff over (replace `PI_IP`):
```
scp display.py run.sh pump_control.py PROGRAM.py GRID_READER.py StripEdgeRefinement.py pi@PI_IP:/home/pi/filterprogram/
scp runs/detect/water_strip_v2/weights/best.pt pi@PI_IP:/home/pi/filterprogram/
```

2. install deps:
```
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 python3-lgpio python3-gpiozero python3-tk
```

3. export model (only for AI mode on Pi):
```
cd /home/pi/filterprogram
python3 -c "from ultralytics import YOLO; YOLO('best.pt').export(format='imx', imgsz=640)"
```
 imx export is picky about OS - if it errors, check ultralytics imx500 notes for supported images.

## running

main ui (waveshare 3.5 / hdmi tk window):
```
./run.sh
# or: python3 display.py
```

grid mode (no AI):
```
python3 GRID_READER.py
python3 GRID_READER.py --calibrate
```

AI mode:
```
python3 PROGRAM.py
```

tests (on pc or pi):
```
python3 test_offline.py
```

## config

timing in `pump_control.py`. gpio pins at top of `pump_control.py` and `PROGRAM.py` / `GRID_READER.py`. calibration in `PROGRAM.py` plus CALIBRATE button in `display.py` writes `calibration_data.json`. `STRIP_ROI` in `display.py` (`WaterFilterApp`) and `GRID_READER.py` is where the camera looks for the strip.
