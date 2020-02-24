import os
import cv2
from PIL import Image
import numpy as np
from numpy import linalg 

def get_thum(image, size=(512, 512), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image

def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(np.average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = np.dot(a / a_norm, b / b_norm)
    return res


def main():
	#img1 = cv2.imread("PXR-0000215.png")
	#img2 = cv2.imread("PXR-0000526.png")
    
        img1 = Image.open("PXR-0000526.png")
        img2 = Image.open("Stanford_hipfx_batch2_004.jpg")
        res = image_similarity_vectors_via_numpy(img1,img2)
        print(res)

if __name__ == "__main__":
        main()
