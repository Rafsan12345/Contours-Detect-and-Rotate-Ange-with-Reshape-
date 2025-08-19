import cv2
import numpy as np

# ইমেজ লোড
img = cv2.imread(r"C:\Users\uiu\Desktop\rice.jpeg")
if img is None:
    raise FileNotFoundError("Image not found! Check the file path.")

img_original = cv2.resize(img, (500, 500))  # মূল ইমেজ resize
img_display = img_original.copy()            # overlay আঁকার জন্য copy

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# থ্রেশোল্ড
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# কনট্যুর বের করা
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# নতুন ইমেজ শুধু rotated contours এর জন্য
rotated_img = np.zeros_like(img)

for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) > 100:   # ছোট noise বাদ
        # মিনিমাম এরিয়া রেক্ট
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # মূল বক্স আঁকা (লাল) মূল ইমেজে
        cv2.drawContours(img_display, [box], 0, (0, 0, 255), 2)

        # কনটুর angle print
        center, size, angle = rect
        print(f"Contour {i+1}: Center={center}, Size={size}, Angle={angle:.2f}°")

        # কনটুর 90° rotate
        M = cv2.getRotationMatrix2D(center, 90, 1.0)

        cnt_2d = cnt.reshape(-1, 2).astype(np.float32)
        rotated_cnt_2d = cv2.transform(np.array([cnt_2d]), M)[0]
        rotated_cnt = rotated_cnt_2d.reshape(-1, 1, 2).astype(np.int32)

        # rotated contour আঁকা (সবুজ) rotated_img এ
        cv2.drawContours(rotated_img, [rotated_cnt], -1, (0, 255, 0), 2)

# resize rotated image
rotated_img_resized = cv2.resize(rotated_img, (500, 500))

# দুই ছবি একসাথে horizontally concatenate
combined = np.hstack((img_original, rotated_img_resized))

cv2.imshow("Original (Left) vs Rotated Contours (Right)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
