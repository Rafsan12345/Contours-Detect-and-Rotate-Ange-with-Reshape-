import cv2
import numpy as np

# --- Image Load ---
img = cv2.imread(r"C:\Users\uiu\Desktop\rice.jpeg")
if img is None:
    raise FileNotFoundError("Image not found! Check the file path.")

# Resize for display
img_original = cv2.resize(img, (500, 500))
img_display = img_original.copy()  # Original contours overlay

# Convert to grayscale and threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# New image for rotated contours
rotated_img = np.zeros_like(img)

for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) > 200:  # Ignore small noise
        rect = cv2.minAreaRect(cnt)
        center, size, angle = rect
        width, height = size

        # Draw original rectangle (red)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(img_display, [box], 0, (0, 0, 255), 2)

        # Determine rotation needed to horizontal
        if width < height:
            # Vertical → rotate to horizontal
            applied_rotation = 90 + angle
        else:
            # Horizontal but not exactly 90 → adjust
            applied_rotation = -angle

        # Normalize rotation within -180 to 180
        if applied_rotation > 180:
            applied_rotation -= 360
        elif applied_rotation < -180:
            applied_rotation += 360

        # Skip rotation if contour is already horizontal ≈ 90°
        if abs(applied_rotation) < 1:  # tolerance 1 degree
            applied_rotation = 0

        print(f"Contour {i+1}: Center={center}, Size={size}, Original Angle={angle:.2f}°, Applied Rotation={applied_rotation:.2f}°")

        # Rotate contour points only if needed
        if applied_rotation != 0:
            M = cv2.getRotationMatrix2D(center, abs(applied_rotation), 1.0)
            cnt_2d = cnt.reshape(-1, 2).astype(np.float32)
            rotated_cnt_2d = cv2.transform(np.array([cnt_2d]), M)[0]
            rotated_cnt = rotated_cnt_2d.reshape(-1, 1, 2).astype(np.int32)
        else:
            rotated_cnt = cnt  # Already horizontal, no change

        # Draw rotated contour (green)
        cv2.drawContours(rotated_img, [rotated_cnt], -1, (0, 255, 0), 2)

        # Write angle on rotated contour
        text_pos = (int(center[0]), int(center[1]))
        cv2.putText(rotated_img, f"{applied_rotation:.1f}°", text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

# Resize rotated image
rotated_img_resized = cv2.resize(rotated_img, (500, 500))

# Concatenate original and rotated images horizontally
combined = np.hstack((img_original, rotated_img_resized))
cv2.imshow("1Original (Left) vs Adjusted Horizontal Contours (Right)", rotated_img)

cv2.imshow("Original (Left) vs Adjusted Horizontal Contours (Right)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
