# Adaptive filtering 
import cv2

def apply_clahe(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    #cv2.imwrite('filtered_image.png', final_img)
    #cv2.imshow('CLAHE Filtered Image', final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

file = apply_clahe('TBX11K/imgs/extra/da+db/train/n11.png')


# Data Augmentation