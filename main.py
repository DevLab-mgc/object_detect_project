import torch
from custom_transforms import get_data_loaders
from model import initialize_model, setup_optimizer_scheduler, save_model#, get_faster_rcnn_model

from engine import train_one_epoch, evaluate#, raw_evaluate, save_metrics
import logging
import argparse
import os
#import smdebug.pytorch as smd # type: ignore

import subprocess
import sys
import gc
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import neptune

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(model, 
         train_loader, 
         val_loader,
         model_dir,
         output_dir,
         hyperparams,
         seed=42):

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    optimizer, lr_scheduler = setup_optimizer_scheduler(model=model,
                                                        learning_rate=hyperparams["lr"],
                                                        weight_decay=hyperparams["weight_decay"],
                                                        momentum=hyperparams["momentum"], 
                                                        schedule_gamma=hyperparams["schedule_gamma"],
                                                        optimizer_type=hyperparams["optimizer"])

   # run = neptune.init_run(mode="offline",project='test')
    run = neptune.init_run(
        project="",   api_token="",
    )
    run["parameters"] = hyperparams
    
    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()


    tot_epochs = hyperparams["epochs"]
    for epoch in range(tot_epochs):
        last_epoch=False
        if epoch-1 == tot_epochs:
            last_epoch=True
        #writer for TensorBoard recording        
        logger.info(f"Starting epoch {epoch + 1}/{tot_epochs}")
        metric_logger = train_one_epoch(
                        model=model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        device=device,
                        epoch=epoch,
                        output_dir=output_dir,
                        print_freq=100,
                        neptune_run=run,
                        scaler=scaler,
                        warmup=hyperparams["warmup"]
                        )
        
        # if we have a failure, break out of loop and end training
        if type(metric_logger)==str:
            print('BREAKING OUT OF TRAINING')
            break


        lr_scheduler.step()

        #  RECORDS TRAINING METRICS
        for meter_name, meter in metric_logger.meters.items():
            run[f"train/epoch/{meter_name}"].append(meter.global_avg)      

        
        metrics = evaluate(model=model, 
                           data_loader=val_loader, 
                           device=device, 
                           epoch=epoch, 
                           output_dir=output_dir,
                          neptune_run=run,
                          last_epoch_flag = last_epoch)
        
        # RECORDING COCO EVALUATION METICS
        stats = metrics.coco_eval['bbox'].stats
        metric_list = ['COCO/AP', 'COCO/AP50', 'COCO/AP75', 'COCO/APs', 'COCO/APm', 'COCO/APl', 'COCO/AR1', 'COCO/AR10', 'COCO/AR100', 'COCO/ARs', 'COCO/ARm', 'COCO/ARl']
        for index, metric_i in enumerate(metric_list):
            run[f"evaluation/epoch/{metric_i}"].append(stats[index])


        # Clear cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Finished epoch {epoch + 1}/{tot_epochs}")


    logger.info("Training completed")
    # Save the trained model
    save_model(model, model_dir, neptune_run=run)
    logger.info("Model Saved")
    run.stop()
    

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    # hyperparameters sent for model selection
    parser.add_argument('--model-name', type=str, default="resnet50_fpn_v2")

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--test-batch-size', type=int, default=4)
    parser.add_argument('--val-batch-size', type=int, default=4)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--schedule-gamma', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--warmup', type=str, default='False')

    parser.add_argument('--use-cuda', type=bool, default=True) #help='TO DISABLE CUDA, PASS EMPTY STRING'
    parser.add_argument('--dry-run', type=bool, default=False) #help='TO ENABLE DRY RUN, PASS ANY NON-EMPTY STRING, OTHERWISE LEAVE EMPTY'

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './model_output')) #os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model_output'))
    parser.add_argument('--increment-model', type=str, default=os.environ.get('SM_CHANNEL_INCREMENT_MODEL', None))

    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA', "./data"))
    parser.add_argument('--annotations', type=str, default=os.environ.get('SM_CHANNEL_ANNOTATIONS',"./ann-files/"))

    parser.add_argument('--anchor-sizes', type=str, default="16,32,64,128,256")

    parser.add_argument('--num-classes', type=int, default=1)

# ADD PARAMETERS FOR EXPERIMENT AND TRAILS
    args, _ = parser.parse_known_args()
    logger.info("Loading Train/Test dataset")
    data_path = args.data    
    output_dir = args.output_data_dir
    annotation_path = args.annotations
    model_dir = args.model_dir

    # if this is local, create directory
    if model_dir == './model_output':
        timestamp = datetime.now().strftime("%d-%m_%H:%M")
        model_dir = f"./{model_dir}/{timestamp}"
        
        os.makedirs(model_dir, exist_ok=True)
        output_dir = model_dir+'/predictions'
        os.makedirs(output_dir, exist_ok=True)

    dry_run = args.dry_run
    
    incremental_model = args.increment_model

    if incremental_model:
        incremental_model= incremental_model+"/model_state_dict.pth"


    hyperparams = {
       "train_batch_size": args.train_batch_size,
        "val_batch_size":args.val_batch_size,
        "epochs":args.epochs,
        "lr":args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "schedule_gamma":args.schedule_gamma,
        "anchor_sizes" : args.anchor_sizes,
        "optimizer" : args.optimizer,
        "warmup" : args.warmup
    }

    #use minimum params to test dry run
    if dry_run:
        hyperparams["train_batch_size"]=2
        hyperparams["val_batch_size"]=2
        hyperparams["epochs"]=2

    
    num_classes = args.num_classes + 1
    #----------------------------------------------MODEL LOAD---------------------------------------------------
    model = initialize_model(num_classes=num_classes, anchor_sizes=hyperparams["anchor_sizes"], model_path=incremental_model)

    #train_input, val_input = 'consolidated_coco.json', 'consolidated_coco.json'
    train_input, val_input = 'train_annotations_BINARY.json', 'val_annotations_BINARY.json'

    #-------------------------------------------CUSTOM TRAINING TRANSFORMS--------------------------------------------------------
    train_annFile, val_annFile = os.path.join(annotation_path,train_input), os.path.join(annotation_path,val_input)

    train_dataset_path, val_dataset_path = data_path, data_path

    train_loader, val_loader = get_data_loaders(train_dataset_path=train_dataset_path,
                                                val_dataset_path=val_dataset_path,
                                                train_annFile=train_annFile,
                                                val_annFile=val_annFile,
                                                train_batch_size=hyperparams["train_batch_size"],
                                                val_batch_size=hyperparams["val_batch_size"],
                                                bbox_in_format='coco',
                                                bbox_out_format='xyxy',
                                                testing=args.dry_run)

    
    logger.info(f"Loaded training dataset with {(train_loader.batch_size)} batch size, {len(train_loader)} batch samples")
    logger.info(f"Loaded validation dataset with {(val_loader.batch_size)} batch size, {len(val_loader)} batch samples")

    #-------------------------------------------DIR TO SAVE PREDICTION--------------------------------------------------------

    print(f'PARAMETERS: \n \t, DRY_RUN:{dry_run}, BATCH_SIZE:{hyperparams["train_batch_size"]}, VAL_BATCH_SIZE: {hyperparams["val_batch_size"]} EPOCHS:{hyperparams["epochs"]} LEARNING_RATE:{hyperparams["lr"]}')

    # #EXECUTE
    main(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          model_dir=model_dir,
          output_dir=output_dir,
          hyperparams=hyperparams)
