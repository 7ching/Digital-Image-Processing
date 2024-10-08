import cv2
import numpy as np

# -----scaling image-----
image = cv2.imread('selfie.jpg')

def scale_image(image, scale_factor, method):
    size_new = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    scaled_image = cv2.resize(image, size_new, interpolation=method)
    return scaled_image

# Bilinear
shrinked_image_bilinear = scale_image(image, 0.12, cv2.INTER_LINEAR)
zoomed_image_bilinear   = scale_image(shrinked_image_bilinear, 7, cv2.INTER_LINEAR)

# Bicubic
shrinked_image_bicubic = scale_image(image, 0.12, cv2.INTER_CUBIC)
zoomed_image_bicubic   = scale_image(shrinked_image_bicubic, 7, cv2.INTER_CUBIC)

# save
cv2.imwrite('bilibear.jpg', zoomed_image_bilinear)
cv2.imwrite('bicubic.jpg',  zoomed_image_bicubic)

# -----calculating quality-----

# MSE (smaller better)
def mse(new_image, ori_image):
    resize_new_image = cv2.resize(new_image, (ori_image.shape[1], ori_image.shape[0]))
    mse_value = np.mean((ori_image - resize_new_image) ** 2)
    return mse_value

# PSNR (bigger better)
def psnr(new_image, ori_image):
    resize_new_image = cv2.resize(new_image, (ori_image.shape[1], ori_image.shape[0]))
    psnr_value = cv2.PSNR(ori_image, resize_new_image)
    return psnr_value

# SSIM (closer to 1 better)
from skimage.metrics import structural_similarity as ssim
def SSIM(new_image, ori_image):
    resize_new_image = cv2.resize(new_image, (ori_image.shape[1], ori_image.shape[0]))

    ori_image_gray = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
    resize_new_image_gray = cv2.cvtColor(resize_new_image, cv2.COLOR_BGR2GRAY)
    
    SSIM_value, dontcare = ssim(ori_image_gray, resize_new_image_gray, full=True)
    return SSIM_value

print(f"----- Bilinear -----")
print(f"MSE  : {mse(zoomed_image_bilinear, image)}")
print(f"PSNR : {psnr(zoomed_image_bilinear, image)}")
print(f"SSIM : {SSIM(zoomed_image_bilinear, image)}")

print(f"----- Bicubic -----")
print(f"MSE  : {mse(zoomed_image_bicubic, image)}")
print(f"PSNR : {psnr(zoomed_image_bicubic, image)}")
print(f"SSIM : {SSIM(zoomed_image_bicubic, image)}")

print(np.mean((zoomed_image_bilinear - zoomed_image_bicubic) ** 2))