import cv2
import numpy as np

# Function to apply Gaussian Blur
def apply_gaussian_blur(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image has been properly loaded
    if image is None:
        print("Error: Could not load image.")
        return

    # Ensure the image is 512x512x3
    if image.shape != (512, 512, 3):
        print("Error: Image is not of the size 512x512x3.")
        return

    # Apply Gaussian Blur with a 5x5 kernel
    # You can change the kernel size and sigma values as needed
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 3)

    # Save the blurred image or you can directly show it using cv2.imshow
    cv2.imwrite('gaussian_blurred_image.png', gaussian_blur)

    # Display the blurred image
    cv2.imshow('Gaussian Blurred', gaussian_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Assuming 'input_image.png' is your 512x512x3 image
apply_gaussian_blur('n11.png')
