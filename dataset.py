import torch
from torchvision.datasets import CocoDetection
from torchvision.ops import box_convert
import cv2
import numpy as np

class ObjectCocoAlbumention(CocoDetection):
    def __init__(self, img_folder, ann_file, processor=None, bbox_in_format='coco', bbox_out_format='xyxy', binary_task=True):
        super(ObjectCocoAlbumention, self).__init__(img_folder, ann_file)
        self.binary_task = binary_task
        self.bbox_in_format = bbox_in_format
        self.bbox_out_format = bbox_out_format
        
        self.default_processor = processor

        if bbox_out_format == 'xyxy':
            self.default_processor.processors['bboxes'].params.format='pascal_voc'
            
    def preprocess(self, img, target):

        if self.bbox_out_format == 'xyxy':
            target['boxes'] = box_convert(target['boxes'], in_fmt="xywh", out_fmt="xyxy")

        transformed = self.default_processor(image=img, bboxes=target['boxes'], category_ids=target['labels'])
        img, bboxes = transformed['image'], transformed['bboxes']

        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        return img, bboxes

    def __getitem__(self, idx):
        img, target = super(ObjectCocoAlbumention, self).__getitem__(idx)
        image_id = self.ids[idx]
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if self.binary_task:
            labels = torch.ones((len(target),), dtype=torch.int64)
        else:
            labels = torch.tensor([ann['category_id'] for ann in target], dtype=torch.int64)

        boxes = [ann['bbox'] for ann in target]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        target_dict = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor([ann['area'] for ann in target], dtype=torch.float32),
            "iscrowd": torch.tensor([ann['iscrowd'] for ann in target], dtype=torch.int64)
        }
        
        img, boxes = self.preprocess(img, target_dict)

        target_dict["boxes"] = boxes.clone().detach()

        return img, target_dict


