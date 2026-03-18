# Water Filter Tester - Pi 5

my water quality testing project. uses a raspi 5 with an IMX500 camera to read 16-pad test strips and score the water. has pumps and solenoids to fill a test container, read the strip, dump the test water, then filter the actual water. theres a 3.5" touchscreen for the ui

## how it works

1. press START on the touchscreen
2. pump fills test container (solenoid routes to test side)
3. strip soaks ~5 sec
4. solenoid opens, dumps test water out
5. camera reads the strip - keeps taking frames until it has enough good ones, throws out bad reads
6. shows score + warnings (lead/mercury weigh 3x more than like fluoride)
7. switches solenoid to filter side, pump runs 100 sec
8. press DISPENSE for clean water

## scoring

0-100 weighted scale, bad stuff counts more

- lead, mercury = 3x weight
- copper, iron, manganese = 2x
- free chlorine, total chlorine, h2s, fluoride = 1.5x
- everything else = 1x

if the waters bad it still filters it just warns you whats wrong

## hardware

- raspberry pi 5
- IMX500 cam (fixed position)
- 3.5" touchscreen
- main pump (gpio 22)
- dispense pump (gpio 23)
- routing solenoid (gpio 24) - test vs filter
- dump solenoid (gpio 25) - drains test container
- cartridge switch (gpio 17)
- status led (gpio 27)

## deps

- python 3.9+
- opencv, numpy
- picamera2, gpiozero
- tkinter
- ultralytics (only for training)

## files

- `touchscreen.py` - main gui, does the whole test/filter/dispense thing
- `pump_control.py` - pump and solenoid control
- `PROGRAM.py` - YOLO + color analysis pipeline
- `GRID_READER.py` - simpler reader, no AI, just fixed crop
- `StripEdgeRefinement.py` - edge cleanup
- `display.py` - old lcd driver (replaced by touchscreen but still works)
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

1. copy stuff over:
```
scp touchscreen.py pump_control.py PROGRAM.py GRID_READER.py StripEdgeRefinement.py display.py pi@<PI_IP>:/home/pi/filterprogram/
scp runs/detect/water_strip_v2/weights/best.pt pi@<PI_IP>:/home/pi/filterprogram/
```

2. install deps:
```
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 python3-lgpio python3-tk
pip3 install ultralytics numpy
```

3. export model (only for AI mode):
```
cd /home/pi/filterprogram
python3 -c "from ultralytics import YOLO; YOLO('best.pt').export(format='imx', imgsz=640)"
```

## running

touchscreen mode (the main one):
```
python3 touchscreen.py
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

tests:
```
python test_offline.py
```

## config

timing stuff is at the top of `pump_control.py`. gpio pins too. calibration values in `PROGRAM.py` need to be measured with your actual lighting setup - theres a calibrate button on the touchscreen for that. `STRIP_ROI` in `touchscreen.py` and `GRID_READER.py` is where the camera looks for the strip
