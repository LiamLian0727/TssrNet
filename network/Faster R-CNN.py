import os
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.util import *
from utils.request import *
from utils.datasetLoad import *

BATCH_SIZE = 4
EPOTH_TIMES = None
Use_Cuda = False
TOKEN = '9f131c8c711d'
DEVICE = torch.device("cuda:0" if Use_Cuda and torch.cuda.is_available() else "cpu")
root = 'D:/deeplearn/Instance/交通标志/Traffic identification/network/dataset/CASIA'
model_save_path = os.path.join('/checkpoint', 'model.pth')

print('DEVICE is : ', DEVICE)
print(f'Load in {root}: ')
print(f"Model will save in {model_save_path}")
img_train, img_test = get_dataset(root)

train_set = CasiaDataset(path=root, image=img_train, h=600, w=600, transform=transforms.ToTensor())
test_set = CasiaDataset(path=root, image=img_test, h=600, w=600, transform=transforms.ToTensor())

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
    for batch_idx, (imgs, class_ids, texts, symbols, arrow_heads, relations, txts) in enumerate(train_loader):
        imgs = list(image.to(DEVICE) for image in imgs)
        targets = []
        for i in range(len(imgs)):
            d = {}
            d['boxes'], d['labels'] = get_all_bboxs_and_labels(texts[i], symbols[i], arrow_heads[i])
            targets.append(d)
        optimizer.zero_grad()
        output = model(imgs, targets)
        loss = output['loss_classifier'] \
               + output['loss_box_reg'] \
               + output['loss_objectness'] \
               + output['loss_rpn_box_reg']
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * BATCH_SIZE, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


test_recall = []
test_precision = []
test_F1 = []


def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, class_ids, texts, symbols, arrow_heads, relations, txts) in enumerate(test_loader):
            imgs = imgs.to(device)
            imgs = list(image.to(DEVICE) for image in imgs)
            targets = []
            for batch in range(len(imgs)):
                d = {}
                d['boxes'], d['labels'] = get_all_bboxs_and_labels(texts[batch], symbols[batch], arrow_heads[batch])
                targets.append(d)
            predictions = model(imgs)
            for batch in range(len(predictions)):
                tp, fp, total_num_gt, total_num_infer = 0, 0, 0, 0
                ious = bbox_overlaps(boxes=predictions[batch]['boxes'].numpy().astype('float64'),
                                     query_boxes=targets[batch]['boxes'].numpy().astype('float64')
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
                recall = 100.0 * tp / len(targets[batch]['labels'] + 1e-8)
                precision = 100.0 * tp / (tp + fp + 1e-8)
                F1 = recall * precision / (recall + precision + 1e-8)
                test_recall.append(recall)
                test_precision.append(precision)
                test_F1.append(F1)
    print(f'\nmRecall: {sum(test_recall) / len(test_recall)}' +
          f'\nmPrecision: {sum(test_precision) / len(test_precision)}' +
          f'\nmF1-Score: {sum(test_F1) / len(test_F1)}')


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                             num_classes=56,
                                                             pretrained_backbone=True).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)
epoch = 120
begin = 0

if os.path.exists(model_save_path):
    checkpoints = torch.load(model_save_path)
    model.load_state_dict(checkpoints['model_state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    StepLR.load_state_dict(checkpoints['StepLR_state_dict'])
    train_loss = checkpoints['loss']
    [test_recall, test_precision, test_F1] = checkpoints['eval']
    begin = int(checkpoints['epoch']) + 1

for i in range(begin, epoch):
    print(f"Epoch: {i} / {epoch}")
    print("Training:")
    train(model, DEVICE, train_loader, optimizer, i)
    print("Testing:")
    test(model, DEVICE, test_loader)
    if (i + 1) % 5 == 0:
        post_to_weixi(TOKEN,
                      'Faster R-CNN demo',
                      f'AutoDL Linux/pytorch{torch.__version__}+{DEVICE}',
                      f'Epoch: {i} / {epoch}' +
                      f'\ntrain Loss: {train_loss[-1]}' +
                      f'\ntest Recall: {sum(test_recall) / len(test_recall)}' +
                      f'\ntest Precision: {sum(test_precision) / len(test_precision)}' +
                      f'\nF1-Score: {sum(test_F1) / len(test_F1)}')
    StepLR.step()
    if EPOTH_TIMES and i == EPOTH_TIMES:
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'StepLR_state_dict':StepLR.state_dict(),
            'loss': train_loss,
            'eval': [test_recall, test_precision, test_F1]
        }, model_save_path)

torch.save(model.state_dict(), model_save_path)