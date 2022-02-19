# coding: utf-8
import os
import torch
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

load_json = json.load(open(os.path.join(root, 'gt', '0', 'road_name_0' + ".json"), 'r', encoding="utf-8"))
print(load_json["texts"][0])


class CasiaDataset(Dataset):
    def __init__(self, path, image, transform=None):
        """
            path: File read path
            image: the list of image's name
            transform: Preprocessing the image
        """
        super(CasiaDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.img = image

    def __getitem__(self, index):
        """
        return: The image and its corresponding annotation information and semantic information
        """

        img_idx = self.img[index]
        img_name, num = img_idx.split('&')
        img_name = os.path.splitext(img_name)[0]
        img = Image.open(os.path.join(self.path, 'img', num, img_name + ".jpg")).convert('RGB')
        if self.transform is not None:
            load_img = self.transform(img)  # 转化tensor类型
        load_json = json.load(open(os.path.join(self.path, 'gt', num, img_name + ".json"), 'r', encoding="utf-8"))
        """subdivision of each item in the json
            路标类别
            class_id = load_json["class_id"] 
            文本信息及bbox
            texts = load_json["texts"] 
            符号信息及bbox
            symbols = load_json["symbols"] 
            箭头信息及bbox
            arrow_heads = load_json["arrow_heads"] 
            关系信息（关联关系association_relation与指向关系pointing_relation）
            relations = load_json["relations"] 
        """
        load_txt = open(os.path.join(self.path, 'semantic_description', num, img_name + ".txt"), 'r', encoding="utf-8")
        return load_img, load_json, load_txt

    def __len__(self):
        return len(self.img)


my_transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
]
)

train_set = CasiaDataset(path=root, image=img_train, transform=my_transform)
test_set = CasiaDataset(path=root, image=img_test, transform=my_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=True, num_workers=0)
