# Session 03

## 1. Reactivate your virtual environment
If you haven't created one yet, please follow the isntructions from [Session 2](../session02/session02.md)

1. Open VS Code and press `CMD + Shift + N` (CTRL + Shift + N on Windows). 

2. Drag your folder into the newly created window.
   
3. Open the Terminal inside VS Code. By default you already should be at the correct folder location.

4. Activate the virtual environment
   ```bash
   source ./Yolo11/bin/activate
   ```
   on Windows
   ```bash
   .\Yolo11\Scripts\activate
   ```

<br><br><br>

## 2. Installing YOLO (Mac and Windows)

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

## 3. Run inference on webcam

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
    cv2.imshow("Yolo11 Webcam", annotated_frame)
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

**On my installation i ran into an error with numpy, that wouldn't let me start the script. If that's the case, you need to install an older version of numpy: `pip install "numpy<2"**

<br><br><br>

## 3.1. Inference on images

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


## 3.2 Inference on Video

Specify the path to the Video File and run the detector.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_file = "videos/Barcelona opera reopens with performance for more than 2000 potted plants.mp4"

results = model(video_file, save=True, conf=0.25)

print("done")
```

<br><br><br>

## 3.3 Run inference on a stream from inseccam.org
Modify your webcam script for this. It can stay mostly the same, but we'll need to use another source. Some streams don't work – they might require a login. Just choose another stream.

1. Get your videostream from insecam.org. Rightclick on a video stream and press 'Copy Image Address'
   
2. Create a variable after your imports
   ```python
   #Paste your URL
   stream_url = 'http://url.mjpg'
   ```

3. Change this line below:
   ```python
   cam = cv2.VideoCapture(stream_url) 
   ```

<details>
<summary>Full Code</summary>

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  

stream_url = 'http://190.210.250.149:91/mjpg/video.mjpg'

cam = cv2.VideoCapture(stream_url) 

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to get Stream.")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO11 Insecam.org", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```
</details>
<br><br><br>

## 3.4 Inference on Twitch Stream
To load a Twitch (or YouTube) videostream, we need to install another library first called `streamlink`

1. Install streamlink and ffmpeg-python (for video encoding). It's possible to install multiple libraries just by separating them with space
   ```bash
   pip install ffmpeg-python streamlink
   ```
2. In the python script, we need to change the imports. Add streamlink and subprocess.
   ```python
    import subprocess
    import streamlink
    ```
3. To correctly implement the stream, we need to write a short function.
   
   1. Define twitch url
        ```python
        TWITCH_URL = 'your_twitch_url
        ```
    2. Get Stream URL function
        ```python
        def get_stream_url(url):
            streams = streamlink.streams(url)
            if 'best' in streams:
                return streams['best'].url
            else:
                raise Exception("Could not retrieve stream.")
        ```
4. Let's put our inference also into function, so we can call it later in a main function.
    ```python
    def run_inference():
        stream_url = get_stream_url(TWITCH_URL)

        # Open video stream with OpenCV
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print("Failed to open Twitch stream.")
            return

        print("Streaming and running inference... Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or frame couldn't be read.")
                break

            # Run inference
            results = model(frame)

            # Draw results on frame
            annotated_frame = results[0].plot()

            # Show frame
            cv2.imshow("YOLO11 on Twitch Stream", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    ```

5. Call the inference function in a main function:
    ```python
    if __name__ == '__main__':
    run_inference()
    ```

<details>
<summary>Full Code</summary>
   
```python
import cv2
import subprocess
from ultralytics import YOLO
import streamlink

# Add Twitch URL
TWITCH_URL = ''

model = YOLO('yolo11n.pt')

def get_stream_url(url):
    
    streams = streamlink.streams(url)
    if 'best' in streams:
        return streams['best'].url
    else:
        raise Exception("Could not retrieve stream.")

def run_inference():
    stream_url = get_stream_url(TWITCH_URL)

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Failed to open Twitch stream.")
        return

    print("Streaming and running inference... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or frame couldn't be read.")
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLO11 on Twitch Stream", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_inference()

```
</details>

<br><br><br>

## 3.5 Inference on Youtube Stream

Same as above, just change `TWITCH_URL` to `YOUTUBE_URL` and add your youtube url.

<br><br><br>


## 4. Finetuning

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
# Apple Silicon (M1 Chip and above)
results = model(frame, device="mps")
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

## 5. Using different datasets

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

## 6. The Flamish Scrollers

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

7. **Run on YouTube Stream**
   
   Combine the Inference on YouTube Script with the script we created above!
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



## 7. Getting the data out of the box with OSC
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

### Sending out data from YOLO11 (With the Flamish Scrollers)

1. Modify the python script to send messages, whenever the classes 'person' and 'cell phone' are detected. 

2. Make the script run in fullscreen with OpenCV.



<br><br><br>

## 8. More Yolo Applications

## 8.1 Blur Boxes (easy)



```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture(0)


blurrer = solutions.ObjectBlurrer(
    show=True,  # display the output
    model="yolo11n.pt",  
    # line_width=2, 
    # classes=[0, 2]
    # blur_ratio=0.5,  # adjust percentage of blur intensity, the value in range 0.1 - 1.0
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = blurrer(im0)

    # print(results")  # access the output


cap.release()

cv2.destroyAllWindows()  # destroy all opened windows
````



## 8.1.1 Person Blur with yolo11-seg (intermediate)

Create a blurring effect that only works on persons. Modify the webcam script for this.

1. In our imports, we need to add `numpy`, this is a very popular library for array operations. For ease of use we can reference it as `np`
    ```python
    import cv2
    import numpy as np
    from ultralytics import YOLO
    ```

2. We need to use the yolo-seg model to get masks
    ```python
    model = YOLO('yolov11n-seg.pt')
    ```

3. Open the webcam – this stays the same
    ```python
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    ```

4. To create a mask of the correct size, we need to find out the dimensions of our frame
    ```python
    height, width = frame.shape[:2]
    ```

5. Feed the Yolo model
    ```python
    results = model(frame, verbose=False)
    ```

6. Create an empty grayscale mask in the size of the video
    ```python
    mask = np.zeros((height, width), dtype=np.uint8)
    ```

7. This function gets the data of the mask created by yolo and converts feeds into the empty grayscale mask we created before.
    ```python
    for result in results:
        if result.masks is not None:
            for seg, cls in zip(result.masks.data, result.boxes.cls):
                if int(cls) == 0:  # class 0 = person
                    seg_mask = (seg.cpu().numpy() * 255).astype(np.uint8)
                    seg_mask_resized = cv2.resize(seg_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    mask = cv2.bitwise_or(mask, seg_mask_resized)
    ```
    
8. Convert the grayscale mask to a color mask - so we can use it with color frames.
    ```python
    mask_3ch = cv2.merge([mask] * 3)
    ```

9. Blur the whole image. (51, 51) is the kernel size, increase to blur more. It only takes odd numbers though.
    ```python
    blurred = cv2.GaussianBlur(frame, (51, 51), 0)
    ````

10. Create the final output by defining that where the mask is white (=255) the regions should be blurred.
    ```python
    result_frame = np.where(mask_3ch == 255, blurred, frame)
    ```

11. Show the combined result
    ```python
    cv2.imshow('YOLO11 - Blurred Persons', result_frame)
    ```

12. Logic to end the script.
    ```python
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ```


<details>
<summary>Full Code</summary>

```python
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov11n-seg.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    results = model(frame, verbose=False)

    mask = np.zeros((height, width), dtype=np.uint8)

    for result in results:
        if result.masks is not None:
            for seg, cls in zip(result.masks.data, result.boxes.cls):
                if int(cls) == 0: 
                    seg_mask = (seg.cpu().numpy() * 255).astype(np.uint8)
                
                    seg_mask_resized = cv2.resize(seg_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    mask = cv2.bitwise_or(mask, seg_mask_resized)

    mask_3ch = cv2.merge([mask] * 3)

    blurred = cv2.GaussianBlur(frame, (101, 101), 0)

    result_frame = np.where(mask_3ch == 255, blurred, frame)

    cv2.imshow('YOLOv8 - Blurred Persons', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
```

</details>



## 9. Download an annotated Training Dataset and train it on your machine

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

    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="yourdatasetfolder/data.yaml", epochs=100, imgsz=640)
    ```
4. Start the training
    ```bash
    python train.py
    ```

5. Wait until finished
6. Navigate to the newly created folder `runs/train/weights`and find `best.pt`
7. Copy best.pt save it to a different location and name it `mytraining.pt`

<br><br><br>

## 10. Label your own dataset 

Use a program like AnyLabelling locally to label your own datasets. This software is open source and completely free:
[AnyLabelling Download Page](https://github.com/vietanhdev/anylabeling/releases)

Or use online annotation tools, either [Roboflow](roboflow.com) or [CVAT](cvat.ai). Both offer a free plan and additionally have useful features like dataset exports in correct formats.

<br><br><br>

## 11. Create a Ultralytics HUB Account and Upload your dataset

[Ultralytics Hub](https://www.ultralytics.com/de/hub)
