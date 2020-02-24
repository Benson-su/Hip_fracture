import os
from skimage.measure import compare_ssim
import cv2
import numpy as np
from PIL import Image


medical_classes={
        "Boston",
#        "JHU",
        "Standford_test1"
        }

classfolders = {
        "hipfx",
        "normal"
        }

def cal_psnr(im1, im2):
    mse = (np.abs(im1 - im2) ** 2).mean()
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr


def get_thum(image, size=(512, 512), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image
#root = "/home/data/standford_data/StanfordPXR_imagebank/train/all/random_test2"

center2_counter = 0

center1="CGMH"
root1="../data/test_ssim/"+center1
for center2 in medical_classes:
    root2="../data/test_ssim/"+center2
    avg = 0
    count = 0
    fp = open(center1+"_"+center2+".csv", 'a')

    classpath1 = root1
    classpath2 = root2

    images1 = os.listdir(classpath1)
    images2 = os.listdir(classpath2)
    for image1 in images1:
        imagepath1 = os.path.join(classpath1, image1)
        img1 = get_thum(Image.open(imagepath1))
        img1 = np.array(img1)
        if len(img1.shape) < 3:
            final_img1 = np.array([img1, img1, img1])
            img1 = final_img1.swapaxes(0,2)

        for image2 in images2: 
             imagepath2 = os.path.join(classpath2, image2)
             img2 = get_thum(Image.open(imagepath2))
             img2 = np.array(img2)
             if len(img2.shape) < 3:
                 final_img2 = np.array([img2, img2, img2])
                 img2 = final_img2.swapaxes(0,2)
             res = cal_psnr(img1, img2)
             avg+=res
             count+=1
             print(count)

    fp.write("avg: "+ str(float(avg/count))+", count:"+str(count))
    fp.close()

