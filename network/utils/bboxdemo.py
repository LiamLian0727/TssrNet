from util import bbox_overlaps
import numpy as np
import torch

if __name__ == '__main__':
    bi = [1,2,7]
    qi = [1,2]
    ious = bbox_overlaps(boxes=torch.FloatTensor([[1., 2., 3., 4.],
                                                  [5., 6., 7., 8.],
                                                  [9., 10., 11., 12.]],
                                                 ).numpy().astype('float64'),
                         query_boxes=torch.FloatTensor([[1.5, 2.5, 3.5, 4.5],
                                                        [5., 6., 7., 8.]]
                                                       ).numpy().astype('float64')
                         )

    print(ious)

    print([max(x) for x in ious])

    a = np.where(ious > 0.5, 1, 0)
    print(a)

    print(sum(sum(a)))


