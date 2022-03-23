import torch
from bbox import bbox_overlaps


def get_ids_and_bboxs(bbox_type):
    ids, bboxs = [], []
    for batch in bbox_type:
        id_batch, bbox_batch = [], []
        if not batch:
            ids.append(batch)
            bboxs.append(batch)
            continue
        for item in batch:
            id_batch.append(item['id'])
            bbox_point = item['bbox']
            # [y_min,x_min,y_max,x_max] -> [x_min,y_min,x_max,y_max]
            bbox_point[0], bbox_point[1] = bbox_point[1], bbox_point[0]
            bbox_point[2], bbox_point[3] = bbox_point[3], bbox_point[2]
            bbox_batch.append(torch.Tensor(bbox_point))
        ids.append(id_batch)
        bbox_batch = torch.stack(bbox_batch, dim=0)
        bboxs.append(bbox_batch)
    return {'ids': ids,
            'bboxs': bboxs}


def get_all_type_bboxs(texts, symbols, arrow_heads):
    return {'texts': get_ids_and_bboxs(texts),
            'symbols': get_ids_and_bboxs(symbols),
            'arrow_heads': get_ids_and_bboxs(arrow_heads)}


dic_symbol = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
              16: 16, 21: 17, 22: 18, 45: 19, 30: 20, 31: 21, 32: 22, 33: 23, 40: 24, 41: 25, 43: 26, 44: 27, 46: 28,
              50: 29, 51: 30, 52: 31, 53: 32, 54: 33, 55: 34, 56: 35, 57: 36, 58: 37, 70: 38, 71: 39, 72: 40, 73: 41,
              76: 42, 79: 43, 80: 44, 100: 45, 200: 46
              }

dic_arrowhead = {1: 47, 2: 48, 3: 49, 4: 50, 5: 51, 6: 52, 7: 53, 8: 54}


def get_all_bboxs_and_labels(texts, symbols, arrow_heads):
    bboxs, labels = [], []
    if texts:
        for i in texts:
            bbox_point = i['bbox']
            # [y_min,x_min,y_max,x_max] -> [x_min,y_min,x_max,y_max]
            bbox_point[0], bbox_point[1] = bbox_point[1], bbox_point[0]
            bbox_point[2], bbox_point[3] = bbox_point[3], bbox_point[2]
            bboxs.append(bbox_point)
            labels.append(55)
    if symbols:
        for i in symbols:
            bbox_point = i['bbox']
            # [y_min,x_min,y_max,x_max] -> [x_min,y_min,x_max,y_max]
            bbox_point[0], bbox_point[1] = bbox_point[1], bbox_point[0]
            bbox_point[2], bbox_point[3] = bbox_point[3], bbox_point[2]
            bboxs.append(bbox_point)
            labels.append(dic_symbol[int(i['class'])])
    if arrow_heads:
        for i in arrow_heads:
            bbox_point = i['bbox']
            # [y_min,x_min,y_max,x_max] -> [x_min,y_min,x_max,y_max]
            bbox_point[0], bbox_point[1] = bbox_point[1], bbox_point[0]
            bbox_point[2], bbox_point[3] = bbox_point[3], bbox_point[2]
            bboxs.append(bbox_point)
            labels.append(dic_arrowhead[int(i['class'])])
    if bboxs:
        bboxs = torch.FloatTensor(bboxs)
        labels = torch.LongTensor(labels)
        return bboxs, labels
    else:
        return torch.FloatTensor([0, 0, 1, 1]), torch.LongTensor(0)

