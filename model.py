import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import torch
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_model(num_classes, anchor_sizes="16,32,64,128,256", model_path=None):
    num_classes = num_classes 

    # RESTNET_50 MODEL
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    anchor_sizes = tuple((int(item),) for item in anchor_sizes.split(','))
    anchor_sizes = ((32, 64, 128, 256, 512),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )

    # Replace the model's RPN anchor generator
    model.rpn.anchor_generator = anchor_generator

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],#, '1', '2', '3'],  # Feature maps from the backbone
        output_size=7,  # Size of the output feature map
        sampling_ratio=2  # Sampling ratio for RoIAlign
    )

    # Set the customized RoI align layer
    model.roi_heads.box_roi_pool = roi_pooler

        # Load saved state if model_path is provided
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading model state from {model_path}")
        model.load_state_dict(torch.load(model_path))

    return model


def setup_optimizer_scheduler(model, learning_rate, weight_decay, momentum, schedule_gamma, optimizer_type='sgd'):

    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_type.lower()=='sgd':
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type.lower()=='adam':
        optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.995), eps=1e-6, weight_decay=weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6, 8], gamma=schedule_gamma)

    return optimizer, lr_scheduler

def save_model(model, model_dir, neptune_run=None):
    logger.info("Saving the model.")
    state_path = os.path.join(model_dir, "model_state_dict.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), state_path)

    path = os.path.join(model_dir, "model.pt")
    torch.save(model.cpu(), path)

    if neptune_run:
        neptune_run["model_checkpoints"].upload(state_path)
        neptune_run["model_checkpoints"].upload(path)
