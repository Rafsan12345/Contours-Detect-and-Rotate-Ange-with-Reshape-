import cv2
import numpy as np

def count_rice_grains(image_path, pixel_per_mm=10):
    """
    Counts the number of isolated rice grains in an image 
    and calculates average area, length, width in mm.
    """

    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image. Please check the file path.")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        grain_count = 0
        areas = []
        lengths = []
        widths = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:   # filter small noise
                grain_count += 1
                areas.append(area)

                # bounding box
                x, y, w, h = cv2.boundingRect(contour)
                lengths.append(h)
                widths.append(w)

                # Draw box
                cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

        if grain_count > 0:
            avg_area_px = np.mean(areas)
            avg_length_px = np.mean(lengths)
            avg_width_px = np.mean(widths)

            # Convert to mm
            avg_area_mm2 = (avg_area_px / (pixel_per_mm**2))
            avg_length_mm = avg_length_px / pixel_per_mm
            avg_width_mm = avg_width_px / pixel_per_mm

            print(f"Total rice grains: {grain_count}")
            print(f"Average Area: {avg_area_px:.2f} px²  (~ {avg_area_mm2:.2f} mm²)")
            print(f"Average Length: {avg_length_px:.2f} px  (~ {avg_length_mm:.2f} mm)")
            print(f"Average Width: {avg_width_px:.2f} px  (~ {avg_width_mm:.2f} mm)")

            # Show result on image
            cv2.putText(image, f"Grains: {grain_count}", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        else:
            print("No rice grains detected!")

        # Show image
        resized = cv2.resize(image, (800, 600))
        cv2.imshow("Detected Rice Grains", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return grain_count

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# ---- Run ----
if __name__ == "__main__":
    image_file = r'C:\Users\DCL\Desktop\Rice image process\1.jpeg'
    count_rice_grains(image_file, pixel_per_mm=10)
