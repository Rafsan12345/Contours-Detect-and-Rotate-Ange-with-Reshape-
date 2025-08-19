import cv2
import numpy as np

# Global list to store points
points = []

def select_point(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")  # Print selected point
        cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 Points", temp_image)

# Load image
image = cv2.imread("p.jpeg")
if image is None:
    raise FileNotFoundError("Image not found!")

# Make a copy to draw points
temp_image = image.copy()
cv2.imshow("Select 4 Points", temp_image)
cv2.setMouseCallback("Select 4 Points", select_point)

cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 4:
    raise ValueError("You must select exactly 4 points!")

# Convert points to float32
pts_src = np.array(points, dtype=np.float32)

# Define destination rectangle (width x height)
width = 400
height = 600
pts_dst = np.array([[0,0], [width,0], [width,height], [0,height]], dtype=np.float32)

# Perspective transform
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
bird_eye_view = cv2.warpPerspective(image, M, (width, height))

# Show result
cv2.imshow("Original Image", image)
cv2.imshow("Bird's Eye View", bird_eye_view)
cv2.waitKey(0)
cv2.destroyAllWindows()
