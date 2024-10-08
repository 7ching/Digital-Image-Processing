import cv2
import numpy as np

image = cv2.imread('../origin.png')

def warpping(image, k1, k2):
    h, w = image.shape[:2]
    camera_matrix = np.array([[w, 0, w/2],
                              [0, h, h/2],
                              [0, 0, 1]], dtype = 'float32')
    dist_coeff = np.array([k1, k2, 0, 0])

    wrapped_image = cv2.undistort(image, cameraMatrix=camera_matrix, distCoeffs=dist_coeff)
    return wrapped_image

# barrel distortion
barrel_distort = warpping(image, 0.5, 0)

# pincushion distortion
pincushion_distort = warpping(image, -0.5, 0)

# save
cv2.imwrite('../results/ex2_barrel.png', barrel_distort)
cv2.imwrite('../results/ex2_pincushion.png', pincushion_distort)

