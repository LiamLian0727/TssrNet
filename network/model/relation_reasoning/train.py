import torch
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import torch.nn.functional as F
import time
import numpy as np

from network.model.relation_reasoning.GAT import GAT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.BoolTensor(data.train_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, mask


g, features, labels, mask = load_cora_data()
model = GAT(g,
            in_feature_dim=features.size()[1],
            hidden_dim=[8,12,16,8],
            out_feature_dim=7,
            num_heads=[2,4,8,2]
            )

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
# g = g.to(DEVICE)
# features = features.to(DEVICE)
# labels = labels.to(DEVICE)
# mask = mask.to(DEVICE)
# model.to(DEVICE)
# main loop
dur = []
print('begin-------------------------------------------------------')
for epoch in range(30):
    if epoch >= 3:
        t0 = time.time()

    logits = model(features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
        epoch+1, loss.item(), np.mean(dur)))