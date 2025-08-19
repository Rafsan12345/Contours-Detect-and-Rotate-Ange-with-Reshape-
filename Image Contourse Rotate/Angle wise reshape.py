import cv2
import numpy as np

# --- ইমেজ লোড ---
img = cv2.imread(r"C:\Users\uiu\Desktop\rice.jpeg")
if img is None:
    raise FileNotFoundError("Image not found! Check the file path.")

# মূল ইমেজ resize
img_original = cv2.resize(img, (500, 500))
img_display = img_original.copy()  # overlay আঁকার জন্য copy

# গ্রেস্কেল + থ্রেশোল্ড
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# contour detect
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# নতুন ইমেজ শুধুমাত্র rotated contours এর জন্য
rotated_img = np.zeros_like(img)

for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) > 200:   # ছোট noise বাদ
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # মূল বক্স (লাল) original ইমেজে
        cv2.drawContours(img_display, [box], 0, (0, 0, 255), 2)

        # angle চেক
        center, size, angle = rect
        print(f"Contour {i+1}: Center={center}, Size={size}, Original Angle={angle:.2f}°")

        # 90° এ adjust করার জন্য rotation calculate
        # যদি ইতিমধ্যেই -90° হয়ে থাকে (90° upright) → কিছু change হবে না
        if abs(angle + 90) < 1:  # 1° tolerance
            rotation_angle = 0
        else:
            rotation_angle = 90 - abs(angle)

        #print(f"Applied Rotation: {rotation_angle:.2f}°")

        an = angle - 90 
        # contour rotate
        if rotation_angle != 0:
            M = cv2.getRotationMatrix2D(center, an, 1.0)
            cnt_2d = cnt.reshape(-1, 2).astype(np.float32)
            rotated_cnt_2d = cv2.transform(np.array([cnt_2d]), M)[0]
            rotated_cnt = rotated_cnt_2d.reshape(-1, 1, 2).astype(np.int32)

            # rotated contour আঁকা (সবুজ) rotated_img এ
            cv2.drawContours(rotated_img, [rotated_cnt], -1, (0, 255, 0), 2)

# resize rotated image
rotated_img_resized = cv2.resize(rotated_img, (500, 500))

# দুটি ছবি একসাথে horizontally concatenate
combined = np.hstack((img_original, rotated_img_resized))

cv2.imshow("Original (Left) vs Rotated Contours (Right)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
