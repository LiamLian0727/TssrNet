import numpy as np
import cv2
import random
import os

# 计算数据集像素的均值方差

root = "D:\\deeplearn\\Instance\\SGG\\Traffic identification\\network\\dataset\\CASIA/img"
img, means, stdevs = [], [], []
h, w = 32, 32
imgs = np.zeros([w, h, 3, 1])
for id in range(13):
    arr = os.listdir(os.path.join(root, str(id)))
    train_num = int(len(arr) * 0.9)
    for file in arr:
        img.append(str(id) + '&' + file)
print(f"Load {len(img)} images")

for i in img:
    id, file = i.split('&')
    img_path = os.path.join(root, id, file)
    print(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (h, w))
    img = img[:, :, :, np.newaxis]
    imgs = np.concatenate((imgs, img), axis=3)

imgs = imgs.astype(np.float32) / 255

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse()
stdevs.reverse()

print(f"normMean = {means}")
print(f"normStd = {stdevs}")
print(f"transforms.Normalize(normMean={means},normStd={stdevs})")
