import cv2
import numpy as np

def count_rice_grains(image_path):
    """
    Counts the number of isolated rice grains in an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        int: The number of rice grains found.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image. Please check the file path.")
            return None

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive thresholding to create a binary image
        # This is more robust to variations in lighting
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours in the binary image
        # Each contour will represent a potential rice grain
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a counter for the grains
        grain_count = 0

        # Loop through the found contours and filter them
        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)

            # Filter out small and large areas to only count rice grains
            # These values (200 and 1000) are examples and might need adjustment
            # based on the image resolution and size of the grains.
            #if 200 < area < 1000:
            if  200<area :
                grain_count += 1
                
                # Optional: Draw the contours on the original image to visualize
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # Display the result
        print(f"Number of rice grains detected: {grain_count}")
        
        # Display the image with contours drawn
        resized = cv2.resize(image, (800, 600))
        cv2.imshow("Detected Grains", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return grain_count

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- How to use the function ---
if __name__ == "__main__":
    # Replace 'path_to_your_image.jpg' with the actual path to your image file
    image_file = r'C:\Users\DCL\Desktop\Rice image process\1.jpeg' 
    count = count_rice_grains(image_file)
