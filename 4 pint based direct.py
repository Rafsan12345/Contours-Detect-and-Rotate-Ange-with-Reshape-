import cv2
import numpy as np

# Load image
image = cv2.imread("p.jpeg")
if image is None:
    raise FileNotFoundError("Image not found!")

# Manual points (source points on original image)
# এখানে আপনাকে ছবিতে চারটি কোণ নির্বাচন করতে হবে
# Top-left, Top-right, Bottom-right, Bottom-left
pts_src = np.float32([[354, 470], [713, 483], [810, 982], [359, 996]])

# Destination points (top-down rectangle)
width = 400
height = 600
pts_dst = np.float32([[0,0], [width,0], [width,height], [0,height]])

# Perspective transform matrix
M = cv2.getPerspectiveTransform(pts_src, pts_dst)

# Warp the image
bird_eye_view = cv2.warpPerspective(image, M, (width, height))

# Show images
cv2.imshow("Original Image", image)
cv2.imshow("Bird's Eye View", bird_eye_view)
cv2.waitKey(0)
cv2.destroyAllWindows()
