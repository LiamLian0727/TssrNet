# coding: utf-8
import os
import cv2
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

root = os.path.abspath(os.path.join(os.getcwd(), "..", 'dataset', 'CASIA'))
img_train = []
img_test = []
for num in range(13):
    arr = os.listdir(os.path.join(root, 'img', str(num)))
    num = int(len(arr) * 0.9)
    for file in arr[:num]:
        img_train.append(str(num) + '&' + file)
    for file in arr[num:]:
        img_test.append(str(num) + '&' + file)
print(len(img_train))
print(len(img_test))


class CasiaDataset(Dataset):
    def __init__(self, path, image, transform=None):
        super(CasiaDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.img = image

    def __getitem__(self, index):
        img_idx = self.img[index]
        img_name, num = img_idx.split('&')
        img_name = os.path.splitext(img_name)[0]
        img = Image.open(os.path.join(self.path, 'img', num, img_name + ".jpg")).convert('RGB')
        if self.transform is not None:
            load_img = self.transform(img)  # 转化tensor类型
        load_json = json.load(open(os.path.join(self.path, 'gt', num, img_name + ".json")))
        # n = len(load_json)
        # bboxes = load_json['annotation'][n]['segmentation']
        load_txt = open(os.path.join(self.path, 'semantic_description', num, img_name + ".txt"), 'r', encoding="utf-8")
        return load_img, load_json, load_txt

    def __len__(self):
        return len(self.img)


my_transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
]
)

trainset = CasiaDataset(path=root, image=img_train, transform=my_transform)
testset = CasiaDataset(path=root, image=img_test, transform=my_transform)
