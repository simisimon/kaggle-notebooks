#!/usr/bin/env python
# coding: utf-8

# ## ConvMixer for object detection using the DETR approach
# It takes forever to converge !
# 
# TODO:
#  - Implementation of the mAP-50
#  - Data augmentation

# In[ ]:


from torchvision import transforms as trsfm
from torchvision.ops._box_convert import _box_cxcywh_to_xyxy as xywh_to_xyxy
from scipy.optimize import linear_sum_assignment as lsa
from torch.nn import functional as F
from PIL import Image
import torchvision
import torch
import json
import time


# In[ ]:


MAX_ITER = 55
BATCH_SIZE = 64
SUBDIVISIONS = 16
NUM_CLASSES = 20
GRAD_NORM = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## Dataset

# In[ ]:


class PascalVOC(torch.utils.data.Dataset):
    def __init__(self, location: str):
        super(PascalVOC, self).__init__()
        self.location = location
        self.classes = open(f'{location}/classes.txt', mode='r').read().splitlines()
        self.items = json.load(open(f'{location}/targets.json', mode='r'))
        self.transforms = trsfm.Compose((
            trsfm.Resize((416, 416)),
            trsfm.ToTensor(),
            trsfm.ConvertImageDtype(torch.float32),
            trsfm.Normalize((0.4564, 0.4370, 0.4081), (0.2717, 0.2680, 0.2810))))

    @classmethod
    def post_process_batch(self, items: list):
        images = torch.stack(tuple(zip(*items))[0])
        targets = tuple(zip(*items))[1]
        return images, targets

    def __getitem__(self, index: int):
        image = Image.open(f"{self.location}/images/{self.items[index]['filename']}").convert('RGB')
        targets = [x['bbox'] + [self.classes.index(x['class'])] for x in self.items[index]['targets']]
        return self.transforms(image), torch.Tensor(targets)

    def __len__(self):
        return len(self.items)


# ## Hungarian matcher (from DETR)

# In[ ]:


def box_iou_union(boxes1, boxes2):
    area1 = torchvision.ops.boxes.box_area(boxes1)
    area2 = torchvision.ops.boxes.box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou_union(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area

class HungarianMatcher(torch.nn.Module):
    def __init__(self) -> None:
        super(HungarianMatcher, self).__init__()

    @torch.no_grad()
    def forward(self, outputs: torch.Tensor, targets: list) -> list:
        B, N, L = outputs.shape
        out_prob = outputs[..., 5:].view(-1, L - 5)
        out_bbox = outputs[..., :4].view(-1, 4)
        tgt_classes = torch.cat([x[:, -1] for x in targets]).long()
        tgt_bbox = torch.cat([x[:, :4] for x in targets])

        cost_class = -out_prob[:, tgt_classes]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1.0)
        cost_giou = -generalized_box_iou(xywh_to_xyxy(out_bbox), tgt_bbox)

        costs = ((cost_bbox * 5) + (cost_giou * 2) + (cost_class * 1)).view(B, N, -1).cpu()
        indices = [lsa(item[idx]) for idx, item in enumerate(costs.split([x.size(0) for x in targets], -1))]
        return [(torch.LongTensor(i), torch.LongTensor(j)) for i, j in indices]


# ## Loss function (from DETR)

# In[ ]:


class ObjectDetectionCriterion(torch.nn.Module):
    def __init__(self, num_classes: int, eos_coef: float = 0.1) -> None:
        super(ObjectDetectionCriterion, self).__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher()
        self.losses = dict({'boxes': 0, 'classes': 0, 'cardinality': 0})
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[self.num_classes] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def zero_losses(self) -> None:
        self.losses.update({'boxes': 0, 'classes': 0, 'cardinality': 0})

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets):
        num_targets = torch.as_tensor([item.size(0) for item in targets], device=DEVICE).float()
        predictions = torch.sum(outputs[..., 4:].argmax(-1) != self.num_classes, dim=1).float()
        return F.l1_loss(predictions, num_targets)

    def loss_labels(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        out_classes = outputs[..., 4:]
        tgt_classes_obj = torch.cat([item[:, -1][i] for item, (_, i) in zip(targets, indices)]).long()
        tgt_classes = torch.full(out_classes.shape[:2], self.num_classes, device=DEVICE).long()
        tgt_classes[idx] = tgt_classes_obj
        return F.cross_entropy(out_classes.transpose(1, 2), tgt_classes, self.empty_weight)

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        bbox_out = outputs[..., :4][idx]
        bbox_tar = torch.cat([item[i, :4] for item, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(bbox_out, bbox_tar, reduction='none')
        loss_giou = torchvision.ops.generalized_box_iou_loss(xywh_to_xyxy(bbox_out), bbox_tar)
        return (loss_bbox.sum().div(num_boxes) * 5) + (loss_giou.sum().div(num_boxes) * 2)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs: torch.Tensor, targets: list):
        targets = [item.to(DEVICE) for item in targets]
        num_boxes = sum([item.size(0) for item in targets])
        indices = self.matcher(outputs, targets)
        loss_boxes = self.loss_boxes(outputs, targets, indices, num_boxes)
        loss_classes = self.loss_labels(outputs, targets, indices)
        loss_cardinality = self.loss_cardinality(outputs, targets)
        self.losses['boxes'] += loss_boxes
        self.losses['classes'] += loss_classes
        self.losses['cardinality'] += loss_cardinality
        return loss_boxes + loss_classes + loss_cardinality


# ## ConvMixer with single detection head

# In[ ]:


class Residual(torch.nn.Module):
    def __init__(self, func) -> None:
        super(Residual, self).__init__()
        self.func = func

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.func(inputs) + inputs
    
class DetectionHead(torch.nn.Module):
    def __init__(self, dim: int, num_classes: int, bbox_attrs: int = 4) -> None:
        super(DetectionHead, self).__init__()
        self.bbox_attrs = bbox_attrs
        self.func = torch.nn.Sequential(
            torch.nn.Conv2d(dim, 96, kernel_size=3, stride=3),
            torch.nn.GELU(),
            torch.nn.GroupNorm(1, 96),
            torch.nn.Conv2d(96, (bbox_attrs + num_classes + 1), kernel_size=1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.func(inputs)
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
        out[..., :self.bbox_attrs] = out[..., :self.bbox_attrs].sigmoid()
        out[..., self.bbox_attrs:] = out[..., self.bbox_attrs:].softmax(-1)
        return out

def ConvMixer(
    in_channels: int,
    dim: int,
    depth: int,
    kernel_size: int = 9,
    patch_size: int = 7,
    out_features: int = 1000
):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, dim, patch_size, stride=patch_size),
        torch.nn.GELU(),
        torch.nn.GroupNorm(1, dim),
        *(torch.nn.Sequential(
            Residual(torch.nn.Sequential(
                torch.nn.Conv2d(dim, dim, kernel_size, groups=dim, padding='same'),
                torch.nn.GELU(),
                torch.nn.GroupNorm(1, dim))),
            torch.nn.Conv2d(dim, dim, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.GroupNorm(1, dim)) for d in range(depth)),
        DetectionHead(dim, out_features))


# ## Training

# In[ ]:


model = ConvMixer(3, 1536, 20, 9, 7, NUM_CLASSES).to(DEVICE)
trainval_set = PascalVOC('/kaggle/input/42aivision')
train_set, test_set = torch.utils.data.random_split(
    dataset=trainval_set,
    lengths=[round(len(trainval_set) * 0.85), round(len(trainval_set) * 0.15)])
train_set = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=(BATCH_SIZE // SUBDIVISIONS),
    collate_fn=PascalVOC.post_process_batch,
    pin_memory=True,
    shuffle=True,
    drop_last=True)
test_set = torch.utils.data.DataLoader(
    dataset=test_set,
    collate_fn=PascalVOC.post_process_batch,
    batch_size=len(test_set))
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    weight_decay=5e-4,
    lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=MAX_ITER,
    eta_min=5e-5)
criterion = ObjectDetectionCriterion(num_classes=NUM_CLASSES).to(DEVICE)
metrics = {'boxes': list(), 'classes': list(), 'cardinality': list()}


# In[ ]:


print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')


# In[ ]:


for epoch in range(1, MAX_ITER + 1):
    criterion.zero_losses()
    optimizer.zero_grad()
    running_loss, start = 0, time.time()
    for minibatch, (images, targets) in enumerate(train_set, 1):
        out = model(images.to(DEVICE))
        loss = criterion(out, targets)
        running_loss += loss.item()
        loss.backward()
        if minibatch % SUBDIVISIONS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
    scheduler.step()
    for key, value in criterion.losses.items():
        metrics[key].append(value.item())
    print(f"epoch {epoch:>2d}/{MAX_ITER:<2d}| loss:{running_loss:.3f}, {(', ').join([f'{k}:{v[-1]:.2f}' for k, v in metrics.items()])}, time:{time.time() - start:.0f}")


# ## Print losses and save model

# In[ ]:


from matplotlib import pyplot as plt
_, axis = plt.subplots(1, 3, figsize=(12, 4))
for index, (key, values) in enumerate(metrics.items()):
    axis[index].plot(values)
    axis[index].set_title(key)
plt.tight_layout()
plt.show()


# In[ ]:


torch.save(model.state_dict(), 'convmixer-1536-20.pth')

