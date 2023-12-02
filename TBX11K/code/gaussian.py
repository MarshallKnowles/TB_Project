import cv2
import numpy as np


def apply_gaussian_blur(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    if image.shape != (512, 512, 3):
        print("Error: Image is not of the size 512x512x3.")
        return

    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 3)

    cv2.imwrite('gaussian_blurred_image.png', gaussian_blur)

    cv2.imshow('Gaussian Blurred', gaussian_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

apply_gaussian_blur('TBX11K/imgs/extra/da+db/train/n11.png')
