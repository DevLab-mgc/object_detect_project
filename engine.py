import math
import sys
import time
import json
import torch
#import torchvision.models.detection.mask_rcnn
import utils
import torchvision
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

import logging
import os
from custom_metrics import evaluate_map
import torch
#import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes
import numpy as np
from PIL import Image

import neptune.new as neptune
from neptune.types import File

import torch
import math
import sys
import utils  # Assuming you have a utils module with the necessary components

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, neptune_run=None, output_dir='', scaler=None, warmup="True", accumulation_steps=4):
    """
    Scaler helps to increase throughput and reduce memory - scales loss to prevent underflow/overflow
    """
    # Pre-epoch memory management
    torch.cuda.empty_cache()

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # CREATES A WARMUP LEARNER FOR THE FIRST EPOCH
    lr_scheduler = None
    if warmup == "True":
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

    tot_images = 0
    optimizer.zero_grad()
    
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        tot_images += len(images)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            return (f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        if scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()

        if (i + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if lr_scheduler is not None and warmup == "True":
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Log gradients to TensorBoard
        # if writer:
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             writer.add_histogram(f"gradients/{name}", param.grad, epoch)
        
        # Post-batch memory management
        #writer.flush()
        del images, targets, loss_dict, losses
        torch.cuda.empty_cache()

    # Ensure optimizer step if total batches are not divisible by accumulation_steps
    if (i + 1) % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    print(f'------------ TOTAL IMAGES IN TRAINING EPOCH {epoch}: {tot_images} ------------')
    return metric_logger



# --------------------------- EVALUATION --------------------------- #

@torch.inference_mode()  # this takes care of the torch.no_grad() needs
def evaluate(model, data_loader, device, epoch=0, output_dir='', neptune_run=None, last_epoch_flag=False):
    # Pre-evaluation memory management
    torch.cuda.empty_cache()

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    predictions = []
    targets_list = []

    # Instantiate the custom loss function
    #custom_loss_fn = CustomValLoss()

    # Initialize running loss variables
    total_batches = 0

    tot_images = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):

        images = list(img.to(device) for img in images)
        tot_images+=len(images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        total_batches += 1

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}


        #CUSTOM METRIC 
       # out = evaluate_map(res, targets)

        evaluator_time = time.time()
        coco_evaluator.update(res)#, gt_trans)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # Save predictions and targets
        predictions.extend(outputs)
        targets_list.extend(targets)

        if neptune_run is not None and last_epoch_flag:# and epoch==(neptune_run['parameters']['epochs']-1):
            plot_save_predictions(images=images, outputs=res, neptune_run=neptune_run)
            
        # Post-batch memory management
        del images, outputs, targets, res
        torch.cuda.empty_cache()

    # Save predictions and targets to disk
    save_predictions_and_targets(predictions, targets_list, epoch, output_dir, 'eval')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    print(f'------------ TOTAL IMAGES IN EVAL EPOCH {epoch}: {tot_images} ------------')

    return coco_evaluator


def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
   # logger.info(f"Metrics saved to {filepath}")


def save_predictions_and_targets(predictions, targets, epoch, output_dir, prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pred_file = os.path.join(output_dir, f"{prefix}_predictions_epoch_{epoch}.json")
    target_file = os.path.join(output_dir, f"{prefix}_targets_epoch_{epoch}.json")
    
    # Convert tensors to lists for JSON serialization
    predictions_serializable = []
    for output in predictions:
        pred_serializable = {k: v.tolist() if torch.is_tensor(v) else v for k, v in output.items()}
        predictions_serializable.append(pred_serializable)
    
    targets_serializable = []
    for target in targets:
        target_serializable = {k: v.tolist() if torch.is_tensor(v) else v for k, v in target.items()}
        targets_serializable.append(target_serializable)

    with open(pred_file, 'w') as f:
        json.dump(predictions_serializable, f)

    with open(target_file, 'w') as f:
        json.dump(targets_serializable, f)

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def plot_save_predictions(images, outputs, neptune_run=None):
    for img, output_key in zip(images, outputs):
        img_id = output_key
        output = ouputs[img_id]
        boxes = output['boxes']
        labels = output['labels']
        img_with_boxes = torchvision.utils.draw_bounding_boxes(img.cpu(), boxes, labels=labels, colors='red', width=2)
        img_with_boxes = torchvision.transforms.ToPILImage()(img_with_boxes)

        # Save image to a bytes buffer
        buffer = io.BytesIO()
        img_with_boxes.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Log image to Neptune
        neptune_run[f"eval/e{epoch}/image_{img_id}.png"].upload(buffer)
