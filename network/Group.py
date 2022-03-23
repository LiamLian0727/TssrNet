import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from network import get_dataset, CasiaDataset, collate_func

root = 'D:\\deeplearn\\Instance\\交通标志\\Traffic identification\\network\\dataset\\CASIA'
print('root: ', root)
img_train, img_test = get_dataset(root)

train_set = CasiaDataset(path=root, image=img_train, h=600, w=600, transform=transforms.ToTensor())
test_set = CasiaDataset(path=root, image=img_test, h=600, w=600, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=2,
                                           shuffle=False,
                                           collate_fn=collate_func,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2,
                                          shuffle=True,
                                          collate_fn=collate_func,
                                          num_workers=0)

RolAlign = torchvision.ops.RoIAlign(output_size=[5, 5], sampling_ratio=-1, spatial_scale=1)

# inputTensor = torch.rand(2, 3, 37, 37)
# #[batch_id, x1, y1, x2, y2]，其中(x1, y1)为左上角，(x2, y2)为右下角
# boxes = torch.rand(2, 4) * 255
# boxes[:, 2:] += boxes[:, :2]
# pooled_features = RolAlign(inputTensor, [boxes])
# print(pooled_features.shape)

for imgs, class_ids, texts, symbols, arrow_heads, relations, txts in train_loader:
    print('imgs:    ', imgs.shape)
    print('class_ids:   ', class_ids.shape)
    print("-------------------------------------------\n")
    for i in range(2):
        # print("imgs[", str(i), "]: \n", imgs[i])

        unloader = transforms.ToPILImage()
        image = unloader(imgs[i])
        plt.imshow(image)
        plt.title('imgs['+str(i)+']')
        plt.pause(0.001)
        print("class_ids[", str(i), "]: ", class_ids[i])
        print("texts[", str(i), "]: \n", texts[i])
        print("symbols[", str(i), "]: \n", symbols[i])
        print("arrow_heads[", str(i), "]: \n", arrow_heads[i])
        print("relations[", str(i), "]: \n", relations[i])
        print("txts[", str(i), "]: \n", txts[i].read())
        print("-------------------------------------------\n")
    break
    # dirc = get_all_type_bboxs(texts, symbols, arrow_heads)
    # # print('class id: ', class_ids, '\n', dirc)
    # print(dirc['texts']['bboxs'].shape)
    break
