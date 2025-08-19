import cv2
import numpy as np

# Load calibration image (image of ruler with 10 mm marking)
img = cv2.imread("new.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 50, 150)

# Detect contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours for visualization
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Suppose you manually choose one contour (largest contour = ruler marking)
c = max(contours, key=cv2.contourArea)

# Get bounding box
x, y, w, h = cv2.boundingRect(c)



cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)






# Pixel length of the ruler marking (say width in pixels)
pixel_length = w  # or h, depending on orientation

# Known real-world length in mm
real_length_mm = 7

# Calculate pixels per mm
pixel_per_mm = pixel_length / real_length_mm

print(f"Pixel Length = {pixel_length} px")
print(f"Real Length = {real_length_mm} mm")
print(f"Pixel per mm = {pixel_per_mm}")

cv2.imshow("Calibration", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
