import cv2
import numpy as np

# ইমেজ পড়া
image = cv2.imread("new.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold দিয়ে binary image
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Contours বের করা
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ধরা যাক 1mm = 14 pixel (আগে measure করা)
pixels_per_mm = 3.5
mm_per_pixel = 1 / pixels_per_mm

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # pixel to mm এ convert
    width_mm = w * mm_per_pixel
    height_mm = h * mm_per_pixel
    area_mm2 = cv2.contourArea(cnt) * (mm_per_pixel ** 2)

    # ইমেজে rectangle আঁকা
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, f"{width_mm:.2f}mm x {height_mm:.2f}mm", 
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    print(f"Object Size: {width_mm:.2f} mm x {height_mm:.2f} mm | Area: {area_mm2:.2f} mm^2")


resized_image = cv2.resize(image, (1000, 1000))
cv2.imshow("Contours", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
