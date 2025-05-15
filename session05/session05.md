# Session 05

## 1. Label your own dataset 

Use a program like AnyLabelling locally to label your own datasets. This software is open source and completely free:
[AnyLabelling Download Page](https://github.com/vietanhdev/anylabeling/releases)

**Mac Users:** If you encounter the error that the App was blocked because it's not from a verified developer: Go to System Settings > Security > Scroll down, find the blocked app window and press `open anyways`
 
Or use online annotation tools, either [Roboflow](roboflow.com) or [CVAT](cvat.ai). Both offer a free plan and additionally have useful features like dataset exports in correct formats.

For this class, we will stick with AnyLabeling. Be aware that it might take a moment to start. 

<br><br><br>

### Train on your face

1. Collect about 50-100 images of your face. Either by taking them at the spot, or by finding them through your Photo Library. Ofc, more is better.

2. Add some photos (about 20% of your dataset) that aren't depicting your face. 

3. Label all of the Photos in Anylabelling containing your face with one class = your_name

4. Export the dataset with Tools > Export Annotations. Choose YOLO (.txt) format and set an output folder. Split your dataset into 80% train and 20% val. (0% test) 

<br><br><br>

## 2. Upload your dataset to our owncloud 

1. Make sure your dataset is in the right format with train and val folders in place and contains a .yaml file.
   
2. Create a folder with your name on our Owncloud `2025S – S05594 – Machine Learning for Artists` in `Datasets_Face`

3. Upload your dataset folders.

<br><br><br>


## 3. DeepFace

We have noticed, that a reliable detection of the face (especially in different environments/surroundings/lighting situations) is only possible with a vast amount of data. Even One-Shot systems like YOLOE fail more or less detecting the same face reliably. 

For face detection we can use [Deepface](https://github.com/serengil/deepface) to detect faces only using a single reference image.

1. Create a new folder on your computer, name it accordingly

2. Open the folder in VS Code and create a virtual environment, as we did with YOLO. **Important**: Make sure you use a Python Version from 3.7-3.10. Newer versions will not work!

3. In session05 navigate to folder `Deepface` and download the requirements.txt.

4. Place the requirements.txt file in your folder 

5. Install requirements with: 
    ```bash
    pip install -r requirements.txt
    pip install deepface
    ```
    

6. Code

    1. Import your libraries
        ```python
        from deepface import DeepFace
        import cv2
        ```

    2. Set your reference image. This is the photo we want to run the detection on.
        ```python
        reference_image_path = "images/foto.jpeg"
        ```

    3. Initialize your Webcam – a true classic.
        ```python
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Could not open webcam.")
            exit()
        ```

    4. Load the Haar Cascade Face Detector. This is a built in function of OpenCV to detect faces. More info [here](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
        ```python
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        ```

    5. Start the main Loop and read frames continuously
        ```python
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break
        ```

    6. This type of face detection works better on B/W images. Convert the frame to grayscale:
        ```python
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ```

    7. Detect Faces on the grayscale image
        ```python
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        ``` 
    
    8. Check if face was found:
        ```python
        if len(faces) > 0:
            x, y, w, h = faces[0]
        ``` 

    9. Write an (temoporary) image from the detected region
        ```python
        detected_face = frame[y:y+h, x:x+w]
        cv2.imwrite("temp_face.jpg", detected_face)
        ```

    10. Compare with the Reference image using the DeepFace library
        ```python
        try:
            result = DeepFace.verify(reference_image_path, detected_face_path, model_name="Facenet")

    11. Check if verified. We define a string and a color for each state.
        ```python
            if result["verified"]:
                label = "Recognized"
                color = (0, 255, 0)  # Green for recognized
            else:
                label = "Unknown"
                color = (0, 0, 255)  # Red for unknown
        except Exception as e:
            label = "Error"
            color = (0, 0, 255)
            print(f"Error during verification: {e}")
        ```

    12. Condition if no faces were detected:
        ```python
        else:
            label = "No Face Detected"
            color = (0, 0, 255)
        ```

    13. Draw rectangle and label
        ```python
        if len(faces) > 0:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        ```

    14. Open the webcam image with the detection.
        ```python
        cv2.imshow("Face Recognition", frame)
        ```

    15. Your classic break the loop conditions
        ```
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
        ```

7. Run the python script.



<details>
<summary>Full Code</summary>

```python
from deepface import DeepFace
import cv2

reference_image_path = "images/foto.jpeg"

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
       
        x, y, w, h = faces[0]
        detected_face = frame[y:y+h, x:x+w]

        detected_face_path = "temp_face.jpg"
        cv2.imwrite(detected_face_path, detected_face)

        try:
            result = DeepFace.verify(reference_image_path, detected_face_path, model_name="Facenet")
            if result["verified"]:
                label = "Recognized"
                color = (0, 255, 0)  # Green for recognized
            else:
                label = "Unknown"
                color = (0, 0, 255)  # Red for unknown
        except Exception as e:
            label = "Error"
            color = (0, 0, 255)
            print(f"Error during verification: {e}")
    else:
        label = "No Face Detected"
        color = (0, 0, 255)


    if len(faces) > 0:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    cv2.imshow("Face Recognition", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

</details>

<br><br><br>

### 3.1 Send an OSC Message when a face is verified

1. Implement the logic from the OSC Send script and send to my ip a string when a face is verified

2. Extra: Also send the position of the face on the frame via OSC.

<br><br><br>

### 3.2 Detect your favourite politician

1. Find an image depicting your (least) favourite politician

2. Detect this person in a YouTube video?


<br><br><br>

### 3.3 The Real Flamish Scrollers!
With the techniques we gathered, we can now write a script that detects a specific person and a phone. For this we need to incorporate the YOLO Logic into our deepface script.

1. Add Ultralytics to your imports. You will need to install it again via `pip install ultralytics`
    ```python
    from deepface import DeepFace
    import cv2
    from ultralytics import YOLO
    ```

2. Define your YOLO model in your main declarations (right after face_cascade)
    ```python
    yolo_model = YOLO("yolo11n.pt")
    ````

3. After the face verification loop (if len(faces)...) add the YOLO Logic.
    ```python 
    results = yolo_model(frame, classes=[67], verbose=False)
    ```
    Then, draw the Yolo Boxes
    ```python
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = yolo_model.model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    ```

<br>

Next, lets add Logic that saves a photo when both of the classes are present in the frame.

4. Just before your DeepFace loop (if len(faces)...), add define a bool. When this bool changes to `true` (and another one for YOLO) the photo will be taken.

5. In the verified if statement add
    ```python
    face_verified = True
    ```

6. The same goes for the YOLO Detection Loop. Before the line `results = ...` add
    ```python
    phone_detected = False
    ```

7. In the loop add 
    ```python
    phone_detected = True
    ```

8. Save a photo – Below your YOLO Detection write a condition that checks if `face_verified` and `phone_detected` are both true

    ```python
    if face_verified and phone_detected:
        cv2.imwrite("detected_face_and_phone.jpg", frame)
    ```

<br><br><br>


## 4. Using Zero Shot Systems like YOLOE

The recent YOLOE Model can detect 4585 classes. It is significantly slower, but can be used to detect more meaningful connections in images. You can find a list of all its classes [here](https://docs.ultralytics.com/models/yoloe/#predict-usage:~:text=predefined%20list%20of%204%2C585%20classes) 

There are three modes available:

### Prompt free

Basically our webcam script, just with a different model:

```python
model = YOLO('yoloe-11s-seg-pf.pt')  
```

<details>
<summary>Full Code<summary>

```python
import cv2
from ultralytics import YOLO

model = YOLO('yoloe-11s-seg-pf.pt')  

cam = cv2.VideoCapture(0) 

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    results = model(frame, device='mps')

    annotated_frame = results[0].plot()

    cv2.imshow('YOLOE', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

</details>

### Text Prompt

```python
from ultralytics import YOLOE
import cv2

model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

names = ["person", "bus"]
model.set_classes(names, model.get_text_pe(names))

cap = cv2.VideoCapture(0)

print("Press '/' to enter new classes (comma-separated). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOE', annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('/'):
        print("Enter new class names, separated by commas (e.g., person,car,dog):")
        cap.release()
        cv2.destroyAllWindows()
        new_classes = input("Classes: ")
        names = [cls.strip() for cls in new_classes.split(",") if cls.strip()]
        if names:
            model.set_classes(names, model.get_text_pe(names))
            print(f"Now searching for: {names}")
        cap = cv2.VideoCapture(0)

cap.release()
cv2.destroyAllWindows()
```

### Visual Prompts

```python
import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

model = YOLOE("yoloe-11s-seg.pt")

reference_images = [
    "images/booth1.jpg" 
]

visual_prompts_list = []


for reference_image_path in reference_images:
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        print(f"Error: Could not load the reference image {reference_image_path}.")
        continue

    
    print(f"Draw the bounding box on {reference_image_path} and press ENTER or SPACE to confirm.")

    bbox = cv2.selectROI(f"Select Bounding Box - {reference_image_path}", reference_image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()


    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    
    visual_prompts_list.append({
        "refer_image": reference_image_path,
        "bboxes": np.array([[x_min, y_min, x_max, y_max]]),
        "cls": np.array([0]),  
    })


cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    
    for visual_prompts in visual_prompts_list:
        results = model.predict(
            frame,
            device="mps",
            refer_image=visual_prompts["refer_image"],  
            visual_prompts={
                "bboxes": visual_prompts["bboxes"],
                "cls": visual_prompts["cls"],
            },
            predictor=YOLOEVPSegPredictor,
        )

        
        annotated_frame = results[0].plot()

    
    cv2.imshow("YOLOE Detection - Specific Person", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()

cv2.destroyAllWindows()
