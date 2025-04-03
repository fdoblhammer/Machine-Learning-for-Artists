## 1. Checking if Python is installed

#### MacOS

1.  Open 'Terminal'
2.  Type `python --version`  

#### Windows

1. Open 'Command Prompt'
2. Type `python --version`  

**If you have python 3.9 installed, you can skip step 2**
<br><br><br>

## 2. Installing Python

#### MacOS

1. Install brew with this Terminal Command: 
    ```bash 
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. Check if brew was installed correctly
    ```bash
    brew --version
    ```

3. Install Python3.9 with brew
    ```bash
    brew install python@3.9
    ```

4. Check if Python3.9 was installed correctly
    ```bash
    python3.9 --version
    ```

5. Set Python3.9 to be the default Python Version
    ```bash
    echo 'alias python3="/usr/local/bin/python3.9"' >> ~/.zshrc
    source ~/.zshrc
    ```

#### Windows

1. Download the Python3.9 installer for Windows(64bit) [here](https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe) 
2. Double click the downloaded installer, make sure to pick version 3.9.13
3. Check "Add Python to PATH" and choose "Customize Installation"
4. Under "Advanced Options" make sure "Add Python to environment variables" is checked. (4th checkbox)
5. Install
6. Verify the installation in Command Prompt/Powershell
   ```bash
   python --version
   ```



### Hello World

Open Terminal or Command Prompt and type:

1. ```python```
2. ```print("Rage Against the Machine Learning")```
3. ```exit()```


<br><br><br>


## 3. Installing YOLO (Mac and Windows)

1. Create a dedicated folder on your machine and open it with Terminal/Command Prompt/Powershell

    *Either* – cd into your folder. You can type `cd + SPACE` and drag your folder into the Terminal/Command Prompt Window
    ```bash
    cd path/to/your/folder
    ```

    *Or* – Right-click on your folder and select `New Terminal at folder...`/`Open in Terminal`

2. Create a virtual environment
    ```bash
    python -m venv Ultralytics
    ````
3. Activate the virtual environment
   ```bash
   source ./Ultralytics/bin/activate
   ```
   on Windows
   ```bash
   .\Ultralytics\Scripts\activate
   ```
4. Install PyTorch
   ```
   pip install torch torchvision torchaudio
    ```
5. Install Ultralytics
   ```
   pip install ultralytics
   ```
6. Install OpenCV
    ```
    pip install opencv-python
    ```

<br>

**Congrats, installation done!**

<br><br><br>

## 4. Run inference on webcam

1. Create a folder on your machine and give it a name e.g `"YOLO11_with_Ferdinand"`
2. Open the folder you just created in your favourite code editor
3. Create a new file and call it `webcam.py`

### webcam.py code breakdown:

Import necessary dependencies:
```python
import cv2
from ultralytics import YOLO
```

Specify which YOLO model you want to use. This points to a "weights"-file (.pt) in your folder and can be interchanged with other weights.
```python
model = YOLO('yolo11n.pt')  
```

We want to use our webcam as a source:
```python
cam = cv2.VideoCapture(0) 
```

Read from the Webcam and print an error if the read fails:
```python
while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break
```

Get results from the YOLO inference:
```python
    results = model(frame)
```

Plot the results so we can see them:
```python
    annotated_frame = results[0].plot()
```

Show the results in a new window:
```python
    cv2.imshow("YoloV8 Webcam", annotated_frame)
```

Break the loop and close the program:
```python
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
````

<br>


<details>
<summary>Full Code</summary>

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  

cam = cv2.VideoCapture(0) 

if not cam.isOpened():
    print("Error: Could not access the webcam.")
    exit()


while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO11 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

</details>

<br><br><br>

## 4.1. Inference on images

#### Breakdown of the code:

These are our imports – additionally to ultralytics we will import **pathlib** which gives us access to the folders on our computer. We don't need opencv here because we are not showing the inference live. 

You might need to install pathlib by executing `pip install pathlib`

```python
from ultralytics import YOLO
from pathlib import Path
```

Reference your YOLO-Model
```python
model = YOLO("yolo11n.pt") 
```

Specify the path to your folder containing the images you want to run the detector on
```python
root_dir = Path("images/")
```

Now we need to make an array list out of the file names to we can iterate on each image.

`str(p)` = function to make a string out of the path

`.glob` = searches through all the file in the folder – use `.rglob` to search all the subfolders too

```python
image_files = [str(p) for p in root_dir.glob("*") if p.is_file()]
```

Next we need a loop that runs the prediction on each file
```python
for image_file in image_files:
    print(f"Processing {image_file}")
    results = model(image_file, save_crop=False, save=True, save_txt=False, conf=0.4)
```
This should create a folder structure like `runs/detect/predict` where all your annotated images will be saved.


<br><br><br>


## 4.2 Inference on Video

Specify the path to the Video File and run the detector.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_file = "videos/Barcelona opera reopens with performance for more than 2000 potted plants.mp4"

results = model(video_file, save=True, conf=0.25)

print("done")
```

<br><br><br>

## 5. Finetuning

**Confidence Threshold**

You can set the confidence threshold to a value between 0 and 1. Detections below will not be shown. Change this line:

```python
results = model(frame, conf=0.5)
```

**IOU Threshold**

Specifies how much the detections can overlap – this is used to eliminate multiple detections on the same object to get a clear output
```python
results = model(frame, iou=0.5)
```

**Image Size**

Sets how the size of the image that will be shown to the detector. Larger images result in slower detections.
```python
results = model(frame, imgsz=1280)
```


**Device**

If you have a graphics card (NVIDIA or Apple M1/2/3/4 Chip) you can run the inference on it. This drastically improves the speed.
```python
#CPU
results = model(frame, device=cpu)
#GPU
results = model(frame, device=0)
```

**Maximum Number of Detections**

Sets the max number of detection. Useful if you just want to detect 1 object.
```python
results = model(frame, max_det=1)
```

 

**Classes**

Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.
```python
results = model(frame, classes=[0])
```

For YOLO11n the classes are listed [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

<br><br><br>

## 6. Using different datasets

To begin, lets try out some other YOLO models from Ultralytics. They will be downloaded automatically when you start the program.

### Official YOLO-Weights from Ultralytics

**Segmentation**
```python
model = YOLO('yolo11n-seg.pt')
```

**TODO**
No background example

**Pose Estimation**
```python
model = YOLO('yolo11n-pose.pt')
```

**Classification**
```python
model = YOLO('yolo11n-cls.pt')
```


### 7. Download an annotated Training Dataset and train it on your machine

Sources:

[Roboflow](https://universe.roboflow.com)
[kaggle](https://kaggle.com)


<br>

1. Download this git repository (as a zip) like described [here](https://medium.com/@bezzam/four-ways-to-download-a-github-repo-a31496ad5b81) under point 1
   (No need to get behind the paywall!)
3. Change directory to the downloaded folder in the macOS/Linux Terminal or Powershell if you are on Windows
4. Create and activate a virtual environment inside the downloaded folder and install Ultralytics. 
   [See here](https://github.com/fdoblhammer/ML-creativecoding/tree/main#3-installing-yolo-mac-and-windows)
5. In the project folder open train.py in your code text editor (VS Code, Sublime Text, etc.)
6. Update the path in "data=" to the absolute path on your system.
   In VS Code you can right-click on the data.yaml file and select "Copy Path", then paste it after "data="
   On macOS drop the data.yaml file into a fresh Terminal window to see the path
8. Don't forget to save train.py after making changes
9. Start the training in your comomand line (Terminal, Powershell, …) by running:
    ```bash
    python train.py
    ```

#### **Optional**
10. Wait until finished
11. Navigate to the newly created folder `runs/train/weights`and find `best.pt`
12. Copy best.pt save it to a different location and name it `mytraining.pt`

<br><br><br>

