import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils.util import *
from utils.request import *
from utils.datasetLoad import *

BATCH_SIZE = 6
TOKEN = '9f131c8c711d'

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'

root = 'D:/deeplearn/Instance/交通标志/Traffic identification/network/dataset/CASIA'
model_save_path = os.path.join('/checkpoint', 'faster_RCNN_model.pth')

print('DEVICE is : ', DEVICE)
print(f'Load in {root}: ')
print(f"Model will save in {model_save_path}")
img_train, img_test = get_dataset(root)

myTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.26798558, 0.3990581, 0.49930063], [0.22402008, 0.21087277, 0.2164303])
])

train_set = CasiaDataset(path=root, image=img_train, h=600, w=600, transform=myTransform)
test_set = CasiaDataset(path=root, image=img_test, h=600, w=600, transform=myTransform)

train_loader = DataLoader(train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          collate_fn=collate_func,
                          num_workers=0)
test_loader = DataLoader(test_set,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         collate_fn=collate_func,
                         num_workers=0)

train_loss = []


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_epoch_loss = []
    for batch_idx, (imgs, _, texts, symbols, arrow_heads, _, _) in enumerate(train_loader):
        imgs = list(image.to(device) for image in imgs)
        targets = []
        for i in range(len(imgs)):
            d = {}
            d['boxes'], d['labels'] = get_all_bboxs_and_labels(texts[i], symbols[i], arrow_heads[i])
            d['boxes'], d['labels'] = d['boxes'].to(device), d['labels'].to(device)
            targets.append(d)
        optimizer.zero_grad()
        output = model(imgs, targets)
        loss = sum(loss for loss in output.values())
        train_epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * BATCH_SIZE, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    train_loss.append(sum(train_epoch_loss) / len(train_epoch_loss))


test_recall = []
test_precision = []
test_F1 = []


def test(model, device, test_loader):
    model.eval()
    test_epoch_recall = []
    test_epoch_precision = []
    test_epoch_F1 = []
    with torch.no_grad():
        for batch_idx, (imgs, _, texts, symbols, arrow_heads, _, _) in enumerate(test_loader):
            imgs = imgs.to(device)
            imgs = list(image.to(DEVICE) for image in imgs)
            targets = []
            for batch in range(len(imgs)):
                d = {}
                d['boxes'], d['labels'] = get_all_bboxs_and_labels(texts[batch], symbols[batch], arrow_heads[batch])
                d['boxes'], d['labels'] = d['boxes'].to(device), d['labels'].to(device)
                targets.append(d)
            predictions = model(imgs)
            for batch in range(len(predictions)):
                tp, fp = 0, 0
                ious = bbox_overlaps(boxes=predictions[batch]['boxes'].numpy().to('cpu').astype('float64'),
                                     query_boxes=targets[batch]['boxes'].numpy().to('cpu').astype('float64')
                                     )
                # iou = [len(boxes),len(query)]
                ious_match = np.where(ious > 0.7, 1, 0)
                for i in range(len(predictions[batch]['boxes'])):
                    for j in range(len(targets[batch]['boxes'])):
                        if ious_match[i][j] == 1:
                            if predictions[batch]['labels'][i] == targets[batch]['labels'][j]:
                                tp += 1
                            else:
                                fp += 1
                recall = 100.0 * tp / (len(targets[batch]['labels']) + 1e-8)
                precision = 100.0 * tp / (tp + fp + 1e-8)
                F1 = recall * precision / (recall + precision + 1e-8)
                test_epoch_recall.append(recall)
                test_epoch_precision.append(precision)
                test_epoch_F1.append(F1)
            if (batch_idx + 1) % 30 == 0:
                print('Text Epoch: {} [{}/{} ({:.0f}%)]\tR : {:.1f}, P : {:.1f}, F1:{:.f}'.format(
                    end_epoch, batch_idx * BATCH_SIZE, len(test_loader.dataset), 100. * batch_idx / len(train_loader),
                    test_epoch_recall[-1], test_epoch_precision[-1], test_epoch_F1[-1])
                )
        test_recall.append(sum(test_epoch_recall) / len(test_epoch_recall))
        test_precision.append(sum(test_epoch_precision) / len(test_epoch_precision))
        test_F1.append(sum(test_epoch_F1) / len(test_epoch_F1))


model = model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 55  # 46+8+1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model = model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.1)
begin_epoch, end_epoch = 0, 50

if os.path.exists(model_save_path):
    checkpoints = torch.load(model_save_path)
    model.load_state_dict(checkpoints['model_state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    StepLR.load_state_dict(checkpoints['StepLR_state_dict'])
    train_loss = checkpoints['loss']
    [test_recall, test_precision, test_F1] = checkpoints['eval']
    begin_epoch = int(checkpoints['epoch']) + 1

for i in range(begin_epoch, end_epoch):
    print(f"Epoch: {i} / {end_epoch}")
    print("Training:")
    train(model, DEVICE, train_loader, optimizer, i)
    print("Testing:")
    test(model, DEVICE, test_loader)
    if (i + 1) % 5 == 0:
        post_to_weixi(TOKEN,
                      'Faster R-CNN demo',
                      'AutoDL Linux+{}'.format(DEVICE),
                      'Epoch:{}/{}, R : {:.1f}, P : {:.1f}'.format(i, end_epoch, test_recall[-1], test_precision[-1])
                      )
    StepLR.step()

torch.save({
    'epoch': i,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'StepLR_state_dict': StepLR.state_dict(),
    'loss': train_loss,
    'eval': [test_recall, test_precision, test_F1]
}, model_save_path)
