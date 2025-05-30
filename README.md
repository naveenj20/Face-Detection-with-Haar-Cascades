# Face Detection using Haar Cascades with OpenCV and Matplotlib

# NAVEEN JAISANKER
# 212224110039
## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame 
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows  

## Program:

```
# (For an existing image)
import cv2
import matplotlib.pyplot as plt
# Load the Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_frontalface_default.xml')
# Load your image
img = cv2.imread('naveen.jpg') # Change to your actual image filename
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
# Draw rectangles around the faces
for (x, y, w, h) in faces:
 cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# Show the result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('oƯ')
plt.title("Detected Faces")
plt.show()

# (For live detection)
import cv2
# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_frontalface_default.xml')
# Start video capture
cap = cv2.VideoCapture(0)
while True:
 ret, frame = cap.read()
 if not ret:
 break
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 faces = face_cascade.detectMultiScale(gray, 1.1, 5)
 for (x, y, w, h) in faces:
 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 cv2.imshow('Face Detection (Press q to quit)', frame)
 if cv2.waitKey(1) & 0xFF == ord('q'):
 break
cap.release()
cv2.destroyAllWindows()
```

## Output:

![Screenshot 2025-05-30 103612](https://github.com/user-attachments/assets/5c275ba7-ff69-4351-a43c-4e537427600f)

![Screenshot 2025-05-30 103632](https://github.com/user-attachments/assets/6b77acad-0ea3-4373-a8f0-5ceb5dca2d93)

## Result:

Thus, the face detection was executed successfully.
