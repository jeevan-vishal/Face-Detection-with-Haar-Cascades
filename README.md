# EXP NO : 12 (Face-Detection-with-Haar-Cascades. )
# NAME : Jeevan Vishal.G.D
# REG NO : 212224240062 

## Aim:

To perform image processing and object detection using ROI segmentation, handwriting detection, and MobileNet-SSD techniques.

## Algorithm:

1. Read the input image using OpenCV and convert it into a suitable color format.
2. Select and extract the required Region of Interest (ROI) using a mask and bitwise AND operation.
3. Convert the image to grayscale and apply Gaussian blur for noise reduction.
4. Detect edges using the Canny edge detection technique.
5. Find and filter contours to identify prominent regions or handwriting.
6. Load the pre-trained MobileNet-SSD model and perform object detection.
7. Display the processed images with segmented regions, detected contours, bounding boxes, and object labels.





## Program :
```py

I) ROI Segmentation in an Image using Bitwise AND
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Read the image and convert the image into RGB
image = cv2.imread('Tamizh.jpeg')  # Replace with your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Step 2: Display the original image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('on')
plt.show()



# Step 4: Set the pixels to display the ROI (Region of Interest)
# Define the coordinates for the Region of Interest (ROI)
# (startY:endY, startX:endX)
roi = image[100:420, 200:550]  # ROI coordinates (adjust as needed)

# Create a blank mask of the same size as the original image
mask = np.zeros_like(image)

# Place the ROI on the mask
mask[100:420, 200:550] = roi
# Step 5: Perform bitwise conjunction of the two arrays using bitwise_and
segmented_roi = cv2.bitwise_and(image, mask)
# Step 6: Display the segmented ROI from the image
segmented_roi_rgb = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2RGB)
plt.imshow(segmented_roi_rgb)
plt.title("Segmented ROI")
plt.axis('off')
plt.show()



II) Handwriting Detection in an Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Read the image and convert it to RGB for displaying
image = cv2.imread('car.png')  # Replace with your actual image file path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
# Original Image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')


# Step 2: Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Step 3: Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Apply Gaussian blur (5x5 kernel)
# Step 5: Use Canny edge detector to find edges
edges = cv2.Canny(blurred_image, 50, 150)  # Detect edges using Canny (thresholds 50 and 150)
# Canny Edge Detection
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')


# Step 6: Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Step 7: Filter contours based on area and draw bounding boxes
result_image = image.copy()  # Create a copy of the original image to draw bounding boxes
for contour in contours:
    if cv2.contourArea(contour) > 50:  # Filter out small areas
        x, y, w, h = cv2.boundingRect(contour)  # Get the bounding box for the contour
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the rectangle
# Handwriting Detection Result
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Handwriting Detection")
plt.axis('off')


III) Object Detection with Labels in an Image using MobileNet-SSD
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Set and add the config_file, weights to your folder
# Ensure you have the MobileNet-SSD files downloaded:
# Download: https://github.com/chuanqi305/MobileNet-SSD
config_file = 'deploy.prototxt'  # Path to the config file
weights = 'mobilenet_iter_73000.caffemodel'  # Path to the weights file
!pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
!pip install opencv-python

!pip uninstall opencv-python -y
!pip install opencv-python==4.10.0.84

import cv2
import sys

print("OpenCV version:", cv2.__version__)
print("Python:", sys.executable)

import sys
!{sys.executable} -m pip uninstall opencv-python -y
!{sys.executable} -m pip install opencv-python==4.10.0.84

# Step 2: Use a pretrained DNN model (MobileNet-SSD v3)
net = cv2.dnn.readNetFromCaffe(config_file, weights)
# Step 4: Create a class label and print the same
class_labels = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable',
                12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep',
                18: 'sofa', 19: 'train', 20: 'tvmonitor'}
# Step 5: Read the image
image = cv2.imread('download.webp')  # Replace with your image path
(h, w) = image.shape[:2]
# Convert image to RGB for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Create a blob for DNN processing
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
# Step 6: Set the model and threshold to 0.5
net.setInput(blob)
detections = net.forward()
# Step 7: Flatten the index, confidence
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:  # Confidence threshold
        index = int(detections[0, 0, i, 1])  # Get class index
        label = class_labels[index]  # Get label name
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
# Step 8: Draw rectangles and labels on the image
        cv2.rectangle(image_rgb, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image_rgb, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# Step 9: Display the image using Matplotlib
plt.imshow(image_rgb)
plt.title("Object Detection with MobileNet-SSD")
plt.axis("off")
plt.show()



```





## Output : 

<img width="710" height="381" alt="ab1" src="https://github.com/user-attachments/assets/dce99c03-5dd7-4adf-a3e5-9eb306d603ee" />




<img width="735" height="397" alt="image" src="https://github.com/user-attachments/assets/93bbc321-f975-4849-a478-1537059fa25d" />


<img width="817" height="471" alt="image" src="https://github.com/user-attachments/assets/f1290e40-a586-4390-9042-47cdd4c785e0" />


<img width="862" height="490" alt="image" src="https://github.com/user-attachments/assets/07ee9b6b-5de1-472a-a70a-566590ebb4e0" />



<img width="362" height="397" alt="image" src="https://github.com/user-attachments/assets/f4e71a2d-93c4-4546-9b22-16ef80e97f73" />




## Result: 
Thus, ROI segmentation, handwriting/region detection, and object detection with labels were successfully performed on the given images.
