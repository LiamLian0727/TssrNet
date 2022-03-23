# coding: utf-8
import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


def get_dataset(path):
    root = path
    img_train = []
    img_test = []
    for id in range(13):
        arr = os.listdir(os.path.join(root, 'img', str(id)))
        train_num = int(len(arr) * 0.9)
        for file in arr[:train_num]:
            img_train.append(str(id) + '&' + file)

        for file in arr[train_num:]:
            img_test.append(str(id) + '&' + file)
    return img_train, img_train


class CasiaDataset(Dataset):
    def __init__(self, path, image, h, w, transform):
        """
            path: File read path
            image: the list of image's name
            transform: Preprocessing the image
        """
        super(CasiaDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.img = image
        self.h = h
        self.w = w

    def __getitem__(self, index):
        """
        return: The image and its corresponding annotation information and semantic information
        """

        img_idx = self.img[index]
        num, img_name, = img_idx.split('&')
        img_name = os.path.splitext(img_name)[0]
        img = Image.open(os.path.join(self.path, 'img', str(num), img_name + ".jpg")).convert('RGB')
        load_img = self.transform(img)  # 转化tensor类型
        c, h, w = load_img.shape
        pad = nn.ZeroPad2d(padding=(0, self.w - w, 0, self.h - h))
        load_img = pad(load_img)
        load_json = json.load(open(os.path.join(self.path, 'gt', str(num), img_name + ".json"), 'r', encoding="utf-8"))
        # subdivision of each item in the json
        # 路标类别
        class_id = load_json["class_id"]
        # 文本信息及bbox
        texts = load_json["texts"]
        # 符号信息及bbox [y_min,x_min,y_max,x_max]
        symbols = load_json["symbols"]
        # 箭头信息及bbox
        arrow_heads = load_json["arrow_heads"]
        # 关系信息（关联关系association_relation与指向关系pointing_relation）
        relations = load_json["relations"]

        load_txt = open(os.path.join(self.path, 'semantic_description', str(num), img_name + ".txt"), 'r',
                        encoding="utf-8")
        return load_img, class_id, texts, symbols, arrow_heads, relations, load_txt

    def __len__(self):
        return len(self.img)


def collate_func(batch):
    imgs = []
    ids = []
    texts_list = []
    symbols_list = []
    arrow_heads_list = []
    relations_list = []
    txt_list = []
    for img, cid, texts, symbols, arrow, relations, txt in batch:
        imgs.append(img)
        ids.append(cid)
        texts_list.append(texts)
        symbols_list.append(symbols)
        arrow_heads_list.append(arrow)
        relations_list.append(relations)
        txt_list.append(txt)
    imgs = torch.stack(imgs, dim=0)
    ids = torch.Tensor(ids)
    return imgs, ids, texts_list, symbols_list, arrow_heads_list, relations_list, txt_list

#
# my_transform = transforms.Compose([
#     transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
# ])
#
# root = os.path.abspath(os.path.join(os.getcwd(), "..", 'dataset', 'CASIA'))
# img_train, img_test = get_dataset(root)
# train_set = CasiaDataset(path=root, image=img_train, h=600, w=600, transform=my_transform)
# test_set = CasiaDataset(path=root, image=img_test, h=600, w=600, transform=my_transform)
# train_loader = torch.utils.data.DataLoader(train_set,
#                                            batch_size=2,
#                                            shuffle=True,
#                                            collate_fn=collate_func,
#                                            num_workers=0)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=2,
#                                           shuffle=True,
#                                           collate_fn=collate_func,
#                                           num_workers=0)
