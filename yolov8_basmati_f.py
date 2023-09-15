import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Load the image
img = cv2.imread('D://amit_2017//BTP//Rice_grain_extraction_from_grid//Grid_images1//image_align//scan_test.jpg')

# Create a copy of the image
img_copy = np.copy(img)

# Convert to RGB so as to display via matplotlib
#img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)



plt.imshow(img_copy)
#plt.show()
# Get user-defined points
print("Click on the 4 corners of the image (in order) start from top left corner and mark in clockwise diretion:")
user_points = plt.ginput(4)

print(len(user_points))

# Convert the points to integer format
user_points = np.array(user_points, dtype=np.int32)

pt_A = user_points[0]
pt_B = user_points[1]
pt_C = user_points[2]
pt_D = user_points[3]

width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))

height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))

# Define input and output points
input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts, output_pts)

# Apply perspective transformation
out = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

# Convert the result to RGB for display
#out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

rotated_out_rgb = cv2.transpose(out)
rotated_out_rgb = cv2.flip(rotated_out_rgb, flipCode=1)


# Display the original and transformed images side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_copy)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(rotated_out_rgb)
plt.title('Transformed Image')

plt.show()

source = cv2.resize(rotated_out_rgb, (1267,1428))

cv2.imshow('Image',source)
cv2.waitKey(0)

model = YOLO('C:\\Windows\\System32\\runs\\detect\\train5\\weights\\best.pt')  # pretrained YOLOv8n model

results = model(source) 

for r in results:
    #  print(r.boxes.xyxy)
    coord_list = r.boxes.xyxy.tolist()
    #  print(r.boxes)

print(type(coord_list[0][0]))
print(len(coord_list))

image = np.copy(source)

rice_type=[]
chalkiness=[]
pos=[]
ar=[]


# Loop through the bounding boxes
for i, (startX, startY, endX, endY) in enumerate(coord_list):
    # Extract the region of interest (ROI)

    startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
    roi = image[startY:endY, startX:endX]
    img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), sigmaX=0, sigmaY=0)
    ret, thresh = cv2.threshold(img_blur, 73, 255, cv2.THRESH_BINARY)

    # Calculate proportion of white pixels
    white_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.size
    white_proportion = white_pixels / total_pixels

    if white_proportion >= 0.17:  # Adjust this threshold as needed
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the index of the largest contour
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]

            x1, y1, w1, h1 = cv2.boundingRect(cnt)
            # Draw bounding box on original image
            cv2.rectangle(image, (startX + x1, startY + y1), (startX + x1 + w1, startY + y1 + h1), (0, 255, 0), 2)
            #cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
            cv2.putText(image, str(i), (startX+x1, startY+y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            aspect_ratio = float(w1) / h1
            ar.append(aspect_ratio)
            
            ret1, thresh1 = cv2.threshold(img_blur, 150, 255, cv2.THRESH_BINARY)
            
            white1 = cv2.countNonZero(thresh1)
            total1 = thresh1.size
            white_proportion1 = white1 / total1
            
            if aspect_ratio > 4.0:
                rice_type.append(f"Basmati_{i}")
                pos.append(coord_list[i])
                # cv2.drawContours(tempimg, [cnt], -1, (255, 0, 0), thickness=cv2.FILLED)
                if white_proportion1 >= 0.02:
                    chalkiness.append(f"chalky_{i}")
                # cv2.putText(tempimg, "chalky", (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 2)  
                #cv2.putText(tempimg, "Basmati", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                   chalkiness.append(f"non-chalky_{i}") 
            else:
                rice_type.append(f"Non-Basmati_{i}")
                pos.append(coord_list[i])
                # cv2.drawContours(tempimg, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)
                if white_proportion1 >= 0.02:
                    chalkiness.append(f"chalky_{i}")
                    # cv2.putText(tempimg, "chalky", (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 2)
                else:
                   chalkiness.append(f"non-chalky_{i}") 
                

        else:
            print("Not enough white pixels at position")

cv2.imshow('Image',image)
cv2.waitKey(0)

cv2.imwrite('D://amit_2017//BTP//Rice_grain_extraction_from_grid//Grid_images1//rotated_yolo1.jpg', image)