## 1. Checking if Python is already installed

To run an Object Detector like YOLO, we need to install the programming language python. Please make sure to follow these steps precicely as it is important to install the correct version with the right configurations.

#### MacOS

1.  Open 'Terminal'
2.  Type `python3 --version`, press Enter
3.  Also try `python --version`

#### Windows

1. Open 'Command Prompt' or 'Powershell'
2. Type `python --version`  

<br>

**This can show 3 different outputs**

1. `No such file or directory` – Python is not installed: You'll need to install python.
2. `Python 3.10.x` or `Python 3.9.x` – Congrats. You already have the right version installed and you can skip the next step. 
3. `Python 2.x.x` or `Python 3.x.x` – You have a different version than 3.10 (or 3.9) installed – You'll need to install 3.10 and tell your machine to use it.
<br><br><br>

## 2. Installing Python 3.10

### MacOS



1. Download the [Python 3.10 Installer for Mac](https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg)

2. Double-click the downloaded .pkg File. If you are on MacOS Sonoma or newer, you might get a warning about an unsupported developer app. To bypass this just **right-click** -> **Open**

3. Follow the installer instructions and wait until the installation finishes.

4. Open the Terminal, check if Python3.10 was installed correctly
    ```bash
    python3 --version
    ```

5. Change/Set Python3.10 to be the default Python Version.
    ```bash
    echo 'alias python3="/usr/local/bin/python3.10"' >> ~/.zshrc
    source ~/.zshrc
    ```
6. On MacOS Catalina and later: 
   ```bash
   open -e ~/.zshrc
   ```

   older Macs (bash):
    ```bash
   open -e ~/.zshrc
   ```

7. Add this at the end of the file:
    ```bash
    alias python=python3
    alias pip=pip3
    ```

8. Save and close the file.

9. Apply changes:
    ```bash
   source ~/.zshrc
   ```

   older Macs (bash):
    ```bash
   source ~/.bash_profile
   ```

10. Check it with `python --version`

<br><br>

### Windows

1. Download the [Python3.10 installer for Windows(64bit)](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe) 
   
2. Double-click the downloaded .pkg File.
   
3. **IMPORTANT:** Check "Add Python to PATH" at the bottom of the Window.
   
4. Install & wait until the installation finishes.
   
5. Verify the installation in Command Prompt/Powershell
   ```bash
   python --version
   ```

    **If this doesnt work, the issue lies mostly within the configuration of the Environment Variables. Here is a [guide](https://readmedium.com/how-to-set-up-a-virtual-environment-with-a-different-python-version-on-windows-10-9900eb0acf9a) on how to set this up correctly, or just ask me.**

<br><br>

### Hello Python

Open Terminal or Command Prompt and type:

1. ```python```
   
2. ```print("Rage Against the Machine Learning")```
   
3. ```exit()```


<br><br><br>

## 3. Creating a virtual environment

There is two ways of working with python: Either with or without an virtual environment. When working with python, in most projects you'll likely need to install *Python Libraries*. These libraries can sometimes interfere with each other, i.e. when you need two different versions of the same library for two different project. For a clean workflow, I recommend to use a Virtual Environment, which can be seen as a glass dome where your python project lives. 

1. Create a dedicated folder for your project on your computer at a location where you can find it again.

2. Open VS Code and press `CMD + Shift + N` (CTRL + Shift + N on Windows). 

3. Drag your folder into the newly created window.
   
4. Open the Terminal inside VS Code. By default you already should be at the correct folder location.

5. Create a virtual environment
    ```bash
    python -m venv Yolo11
    ````
6. Activate the virtual environment
   ```bash
   source ./Yolo11/bin/activate
   ```
   on Windows
   ```bash
   .\Yolo11\Scripts\activate
   ```

In your Command Line Window should now have (Yolo11) prepended. This indicates you are working within the Virtual Environment. If you close your Command Line Window you will need to reactivate the environment with the command above.

<br><br><br>

## 4. Python Basics

### Creating and running python script in CLI

1. Lets create a file in VS Code. Press `CMD + N` or `CTRL + N`(Windows)

2. Save this file with the `.py` extension. Name it e.g. `test.py`

3. To run this type into the CLI and press enter: 
    ```bash
    python test.py
    ```
4. If your file is in a subfolder you'll need to:
    ```bash
    python foldername/file.py
    ````

<br>

### Python Syntax Basics

Put this into your test.py file

1. **Printing** something to the console works like this
    ```python
    print("Rage Against the Machine Learning)
    ```

2. We use **variables** to store different kind of data
   ```python
   # Numbers
   x = 5
   x = 0.1
   ```
   ```python
   # Strings
   name = "Ferdinand"
   ```
   ```python
   # Booleans
   active = True
   ```

3. **Comments** are used to make your code more readable aswell as to deactivate lines you don't need.
   ```python
   # One Line Comment

   """ 
   Multi 
   Line 
   Comment
   """
   ```

4. **If/else statements** and **loops** 
    <br>if/else
    ```python
    rage = 100
    if rage >= 100:
        print(rage)
    else:
        print("i sleep")
    ```

    for loops
    ```python
    for i in range(500):
        print(i)
    ```

    while loops
    ```python
    rage = 0
    while rage < 100:
        print(rage)
        rage += 5

5. **Functions** make your code reusable
    ```python
    rage = 100

    def learning(rage):
        print(f"Learning: {rage}%")

    learning(rage)
    ```

6. **Lists** can store more than one Value
    ```python
    learning = ["Rage", "against", "the", "Machine"]
    print(learning[1])

    #print(learning[0:4])
    ```

7. **Dictionaries** 
    ```python
    progress = {"Rage": "Yes", "Learning": 100}
    print(progress["Rage"])
    ```

8. Working with **libraries**
   ```python
   import time 

   rage = 0

    while rage < 100:
        print(f"{rage} %")
        rage += 5
        time.sleep(1)
    ```

    The `time` library should come by default with the Python installation, but many libraries that extend Pythons functionality need to be installed first. Make sure your virtual environment is active, then run this in the CLI:
    ```bash
    pip install examplelibrary
    ```

9.  If a python script stops working, but doesn't exit itself, press `CRTL + C` to stop it from the CLI

<br><br><br>

## 5. OpenCV
OpenCV is a library used mainly for real-time computer vision. We will use it to display images and videos and to do some manipulations on them.

### Installation

In your CLI type this an press enter to install opencv and wait until it finishes (takes a moment):
```bash
pip install opencv-python==4.10.0.84
```

At the moment writing, the newest opencv didn't work for me, so we will download version 4.10.0.84

### Load and display in image
1. Find an image on your machine, create a new folder in VS Code called `ìmages` and drag it inside.
2. Create a new python script and name it ?image.py`
3. Import opencv
    ```python
    import cv2
    ```
4. Load image in folder
    ```python
    image = cv2.imread('images/your_image.jpg')
    ```
5. Show image
    ```python
    cv2.imshow('Image Window', image)
    ```
6. Close if any key is pressed
    ```python
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

### Draw Shapes
1. Rectangle
    ```python
    cv2.rectangle(image, (50, 50), (200, 200), (0, 0, 255), 2)
    #Color is in BGR!
    ```
  
2. Circle
    ```python
    cv2.circle(image, (300, 300), 40, (255, 0, 0), -1)
    ```
3. Results
    ```python
    cv2.imshow('Image Window', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

### Resizing Images
1.  Values are in px
    ```python
    resized_image = cv2.resize(image, (300, 300))

    cv2.imshow('Image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

### Full Screen
1. Press any key to exit
    ```python    
    image = cv2.imread('your_image.jpg')

    cv2.namedWindow('FullScreenWindow', cv2.WINDOW_NORMAL)

    cv2.setWindowProperty('FullScreenWindow', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('FullScreenWindow', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

### Convert the image to Grayscale
1. Some simple image manipulations
   ```python
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Grayscale', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

2. Try out some other manipulations from this table


    | **Action** | **Command** | **Description** |
    |:---|:---|:---|
    | **Grayscale** | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | Convert to black & white. |
    | **Resize** | `cv2.resize(img, (width, height))` | Resize to new dimensions. |
    | **Rotate** | `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` | Rotate 90°, 180°, etc. |
    | **Flip** | `cv2.flip(img, flipCode)` | Flip vertically, horizontally, or both (`flipCode = 0, 1, -1`). |
    | **Blur** | `cv2.GaussianBlur(img, (5,5), 0)` | Soft blur (good for smoothing noise). |
    | **Edge Detection** | `cv2.Canny(img, 100, 200)` | Find edges in the image. |
    | **Change Brightness/Contrast** | `img_new = cv2.convertScaleAbs(img, alpha=1.5, beta=50)` | Adjust contrast and brightness. |
    | **Thresholding (Binarization)** | `_, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)` | Convert to pure black and white. |
    | **Draw Text** | `cv2.putText(img, 'Hello', (50,50), font, 1, (255,255,255), 2)` | Write text onto the image. |
    | **Color Space Change** | `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` | Convert BGR to HSV color space. |
    | **Crop Image** | `cropped = img[y1:y2, x1:x2]` | Slice part of an image (crop). |

### Webcam Video

This chooses the first (0) video device on your computer available. Try to do some image manipulations from above on it.

1. Define the webcam as your source:
    ```python
    cam = cv2.VideoCapture(0) 
    ```
2. Try to read images from the webcam
    ```python
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
    ```

3.  Show the Video (still in the loop!)
    ```python
        cv2.imshow('Webcam', frame)
    ```

4.  Exit when the key 'q' is pressed
    ```python
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ```

5. Release the cam and close window (outside loop)
    ```python
    cap.release()
    cv2.destroyAllWindows()
    ```

### Read an online Video Stream from insecam.org
    ```python
    import cv2

    # Replace with the actual stream URL from insecam.org
    stream_url = 'http://YOUR_STREAM_IP/mjpg/video.mjpg'

    cap = cv2.VideoCapture(stream_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get stream.")
            break

        cv2.imshow('Inseccam.org', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ```



## 6. Installing YOLO (Mac and Windows)

1. Install PyTorch
   ```
   pip install torch torchvision torchaudio
    ```
2. Install Ultralytics
   ```
   pip install ultralytics
   ```

<br>

**Congrats, installation done!**

<br><br><br>

## 7. Run inference on webcam

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

## 7.1. Inference on images

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


## 7.2 Inference on Video

Specify the path to the Video File and run the detector.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_file = "videos/Barcelona opera reopens with performance for more than 2000 potted plants.mp4"

results = model(video_file, save=True, conf=0.25)

print("done")
```

<br><br><br>

## 8. Finetuning

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

## 9. Using different datasets

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

<br><br><br>

## 10. The Flamish Scrollers

A simplified reconstruction of Dries Depoorters work. We'll modify our Python script to print a message when both the classes `person` and `cell phone` are being detected.

We will modify our webcam script, if you need the code its here:
<details>
<summary>Code Webcam.py</summary>

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

1. **Find out the number of the classes we want to detect**

    For YOLO11n the classes are listed [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

    <details>

    <summary>Classes</summary>

    `person: 0`
    `cell phone: 67`

    </details>


2. **Set the detector to only detect these two classes**

    Change this line:

    ```python
    results = model(frame, classes=[0, 67])
    ```

3. **Unclutter your print**

    The YOLO detector should now only detect the set two classes. Now we want to get a message in the print if both classes are seen. But YOLO is already printing lots of messages – let's suppress those first

    Change this line:

    ```python
    results = model(frame, classes=[0, 67], verbose=False)
    ```

4. **Find the relevant data**

    We now we want to find where to talk to the yolo detector. We can try to find the relevant strings by just printing out to the console.

    ```python
    print(results)
    ```
    narrowing down our search we find out that results is actually an array list so we need to adress it like this `results[0]

    ```python
    print(results[0].boxes)
    ```
    on a further look we find the `cls` value (=class). Lets print this then:

    ```python
    print(results[0].boxes.cls)
    ````


5. **Put the data in the right place (another list)**



    after the line with `results` add this:

    ```python
    detected_classes = set()
    if results[0].boxes is not None:
        for item in results[0].boxes.cls:
            detected_classes.add(int(item))
    ```

    This creates the variable `detected_classes`. If results from the detector are coming in, we look for the fifth value of the detection box – which is the class number – and stores it to `detected_classes`

    If you want to know how a for loop works check [this](https://www.w3schools.com/python/python_for_loops.asp) for reference.

6. **Print if both classes are detected**

    This condition triggers the print if both classes `0` and `67` are detected.

    ```python
    if 0 in detected_classes and 67 in detected_classes:
        print("Stay focused!")
    ```
<br>

<details>

<summary>Full Code</summary>

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  

confidence_threshold = 0.1

cam = cv2.VideoCapture(0) 

if not cam.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    results = model(frame, conf=confidence_threshold, classes=[0, 67], verbose=False)

    #print(results)
    #print(results[0].boxes)
    #print(results[0].boxes.cls)

    detected_classes = set()
    if results[0].boxes is not None:
        for item in results[0].boxes.cls:
            detected_classes.add(int(item))

    
    if 0 in detected_classes and 67 in detected_classes:
        print("Stay focused!")
    
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO11 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

```

</details>


<br><br><br>

## 11. Getting the data out of the box with OSC
Open Sound Control (OSC) is a protocol for sending simple data strings over the network. It is super easy to use and supported in a wide range of programs/languages. You can either send messages locally on your device, or send them to other machines. The devices have to be on the same network though.

Before starting, make sure to install python-osc with:
```bash
pip install python-osc
```

### Simple Sender Script

1. Import necessary libraries
    ```python
    from pythonosc import udp_client
    import time
    ```

2. Define which at IP address you want to send the message to:
   1. Local (Your computer)
        ```python
        ip = "127.0.0.1"
        ```
    1. Everone on the network
        ```python
        ip = "255.255.255.255"
        ```
    2. Find out the specific ip address of the receiving device
        ```python
        ip = "xxx.xxx.xxx.xxx"
        ```
3. Define the port on both devices
    ```python
    port = 5005
    ```

4. Set up the Client
   ```python
   client = udp_client.SimpleUDPClient(ip, port)
   ```
5. Set and send the message
    ```python
    message = "Hello from PythonOSC!"
    client.send_message("/test", message)

    print(f"Sent: {message}")
    ```

### Simple Receiver Script

```python
from pythonosc import dispatcher
from pythonosc import osc_server

def handle_message(address, *args):
    print(f"Received message at {address}: {args[0]}")

# Set up dispatcher
disp = dispatcher.Dispatcher()
disp.map("/test", handle_message)


ip = "127.0.0.1"  
port = 5005       # must match sender
server = osc_server.BlockingOSCUDPServer((ip, port), disp)

print(f"Listening on {ip}:{port}")
server.serve_forever()
```

### Sending out data from YOLO11



<br><br><br>

## 12. Download an annotated Training Dataset and train it on your machine

### What does a training set look like?
- Consists of a lot of representative images of the objects/things the algorithm should detect.
- Normally, results start to get good at 800+ images per class (=type of object)
- For each image, there is a corresponing 'label' file, which holds the information on the class (=what object) and its position on the image (=coordinates)
- The structure of a YOLO Dataset typically looks like this:
- - A folder `train` containing 80% of the files:
- - - folder `images` with image files (.jpg, .png)
- - - folder `labels` with corresponding annotation files (.txt)
- - A folder `val` containing 20% of the files:
- - - folder `images` with image files (.jpg, .png)
- - - folder `labels` with corresponding annotation files (.txt) 
- - A `.yaml`file containing information about our classes and folder location

<br><br><br>

### Download an annotated Training Dataset and train it on your machine

Sources:

[Roboflow](universe.roboflow.com)
[kaggle](kaggle.com)

A not very sophisticated dataset:
[Open/Close Eyes Dataset](https://universe.roboflow.com/isee-gufmk/eyes-zefum/dataset/6)

<br>

1. Extract downloaded dataset folder into your project folder.
2. In your project folder create a file named `train.py`
3. Code:
    ```python
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="yourdatasetfolder/data.yaml", epochs=100, imgsz=640)
    ```
4. Start the training
    ```bash
    python train.py
    ```

    You will probably encounter an error when running this because 

5. Wait until finished
6. Navigate to the newly created folder `runs/train/weights`and find `best.pt`
7. Copy best.pt save it to a different location and name it `mytraining.pt`

<br><br><br>

### Label your own dataset 

Use a program like AnyLabelling locally to label your own datasets. This software is open source and completely free:
[AnyLabelling Download Page](https://github.com/vietanhdev/anylabeling/releases)

Or use online annotation tools, either [Roboflow](roboflow.com) or [CVAT](cvat.ai). Both offer a free plan and additionally have useful features like dataset exports in correct formats.
