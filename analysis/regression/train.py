import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from FrameDatamodule import TyphoonDataModule
from LightningRegressionModel import LightningRegressionModel
from pytorch_lightning.callbacks import ModelCheckpoint

import config
from argparse import ArgumentParser

from datetime import datetime
import torch

# IMAGE_DIRS = ['/dataset/0/wnp/image/',
#               '/dataset/1/wnp/image/', '/dataset/2/wnp/image/']
# METADATA_DIRS = ['/dataset/0/wnp/metadata/',
#                  '/dataset/1/wnp/metadata/', '/dataset/2/wnp/metadata/']
# METADATA_JSONS = ['/dataset/0/wnp/metadata.json',
#                   '/dataset/1/wnp/metadata.json',
#                   '/dataset/2/wnp/metadata.json']
IMAGE_DIRS = ['/dataset/0/wnp/image/',
              '/dataset/1/wnp/image/']
METADATA_DIRS = ['/dataset/0/wnp/metadata/',
                 '/dataset/1/wnp/metadata/']
METADATA_JSONS = ['/dataset/0/wnp/metadata.json',
                  '/dataset/1/wnp/metadata.json']
start_time_str = str(datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))

def custom_parse_args(args):
    """Argument parser, verify if model_name, device, label, size and cropped arguments are correctly initialized"""

    args_parsing = ""
    if args.model_name not in ["resnet18", "resnet50","resnet101", "vgg"]:
        args_parsing += "Please give model_name among resnet18, 50, 101 or vgg\n"
    if args.size not in ["512", "224", 512, 224]:
        args_parsing += "Please give size equals to 512 or 224\n"
    if args.cropped not in ["False", "True", "false", "true", False, True]:
        args_parsing += "Please give cropped equals to False or True\n"
    if torch.cuda.is_available() and int(args.device) not in range(torch.cuda.device_count()):
        args_parsing += "Please give a device number in the range (0, %d)\n" %torch.cuda.device_count()
    if args.labels not in ["wind", "pressure"]:
        args_parsing += "Please give size equals to wind or pressure\n"

    if args_parsing != "": 
        print(args_parsing)
        raise ValueError("Some arguments are not initialized correctly")
    
    if args.size == '512' or args.size == 512:
        args.size = (512,512)
    elif args.size == '224'  or args.size == 224:
        args.size = (224, 224)
    
    if args.cropped in ["False", "false"]: args.cropped = False
    if args.cropped in ["True", "true"]: args.cropped = True

    if args.device == None:
        args.device = config.DEVICES
    else:
        args.device = [int(args.device)]

    return args

def train(hparam):
    """Launch a training with the PytorchLightning workflow, the arguments given in the python command and the hyper parameters in the config file"""
    hparam = custom_parse_args(hparam)

    logger_name = hparam.labels + "_" + hparam.model_name + "_" + str(hparam.size[0])
    if hparam.cropped: logger_name += "_cropped"
    else : logger_name += "_no-crop"

    logger = TensorBoardLogger(
        save_dir="results",
        name= logger_name,
        default_hp_metric=False,
    )

    # Log all hyper parameters
    logger.log_hyperparams({
        'start_time': start_time_str,
        'LEARNING_RATE': config.LEARNING_RATE,
        'BATCH_SIZE': config.BATCH_SIZE,
        'NUM_WORKERS': config.NUM_WORKERS,
        'MAX_EPOCHS': config.MAX_EPOCHS,
        'WEIGHTS': config.WEIGHTS, 
        'LABEL' : hparam.labels,
        'SPLIT_BY': config.SPLIT_BY, 
        'LOAD_DATA': config.LOAD_DATA, 
        'DATASET_SPLIT': config.DATASET_SPLIT, 
        'STANDARDIZE_RANGE': config.STANDARDIZE_RANGE, 
        'DOWNSAMPLE_SIZE': hparam.size, 
        'CROPPED': hparam.cropped,
        'NUM_CLASSES': config.NUM_CLASSES, 
        'ACCELERATOR': config.ACCELERATOR, 
        'DEVICES': hparam.device, 
        'DATA_DIR': config.DATA_DIR, 
        'MODEL_NAME': hparam.model_name,
        # "Scheduler": "Start from 0.001 and divide by 10 every 15epochs"
        })

    # Define globals for multi-directory loading
    global IMAGE_DIRS, METADATA_DIRS, METADATA_JSONS
    
    print(f"Data sources: {len(IMAGE_DIRS)} image dirs, {len(METADATA_DIRS)} metadata dirs, {len(METADATA_JSONS)} metadata JSONs")
    
    # Automatically determine in_channels from the number of directories
    in_channels = hparam.in_channels if hasattr(hparam, 'in_channels') else 3  # Default
    auto_channels = False
    single_channel = False
    
    # Determine channels from directory count if force_channels is not set
    force_channels = getattr(hparam, 'force_channels', False)
    if not force_channels:
        if IMAGE_DIRS:
            if len(IMAGE_DIRS) == 1:
                # Single directory = single channel
                in_channels = 1
                single_channel = True
                auto_channels = True
            elif len(IMAGE_DIRS) > 1:
                in_channels = len(IMAGE_DIRS)
                auto_channels = True
        elif METADATA_DIRS:
            if len(METADATA_DIRS) == 1:
                in_channels = 1
                single_channel = True
                auto_channels = True
            elif len(METADATA_DIRS) > 1:
                in_channels = len(METADATA_DIRS)
                auto_channels = True
        elif METADATA_JSONS:
            if len(METADATA_JSONS) == 1:
                in_channels = 1
                single_channel = True
                auto_channels = True
            elif len(METADATA_JSONS) > 1:
                in_channels = len(METADATA_JSONS)
                auto_channels = True

    # Make sure in_channels is at least 1
    in_channels = max(1, in_channels)
    
    if single_channel:
        print(f"Single directory detected - using 1 channel")
    elif auto_channels:
        print(f"Multiple directories detected - using {in_channels} channels")
    else:
        print(f"Using manually specified in_channels = {in_channels}")
    
    # Set up dataset
    data_module = TyphoonDataModule(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        labels=hparam.labels,
        split_by=config.SPLIT_BY,
        load_data=config.LOAD_DATA,
        dataset_split=config.DATASET_SPLIT,
        standardize_range=config.STANDARDIZE_RANGE,
        downsample_size=hparam.size,
        cropped=hparam.cropped,
        image_dirs=IMAGE_DIRS,
        metadata_dirs=METADATA_DIRS,
        metadata_jsons=METADATA_JSONS
    )
    
    # Setup data module (loads train/val/test datasets)
    try:
        data_module.setup(stage="fit")
        
        # Print dataset sizes for debugging
        train_size = len(data_module.train_dataset) if hasattr(data_module, 'train_dataset') and data_module.train_dataset is not None else 0
        val_size = len(data_module.val_dataset) if hasattr(data_module, 'val_dataset') and data_module.val_dataset is not None else 0
        test_size = len(data_module.test_dataset) if hasattr(data_module, 'test_dataset') and data_module.test_dataset is not None else 0
        
        print(f"Dataset sizes - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
        
        if train_size == 0:
            print("WARNING: Training dataset is empty! Check your data paths and dataset loading.")
            if len(IMAGE_DIRS) == 0 and len(METADATA_DIRS) == 0 and len(METADATA_JSONS) == 0:
                print("ERROR: No data directories or JSON files were provided.")
    except Exception as e:
        print(f"Error setting up data module: {str(e)}")
        import traceback
        traceback.print_exc()

    # Model selection
    regression_model = LightningRegressionModel(
        learning_rate=config.LEARNING_RATE,
        weights=config.WEIGHTS,
        num_classes=config.NUM_CLASSES,
        model_name=hparam.model_name,
        in_channels=in_channels  # Use the determined number of channels
    )
    # regression_model = regression_model.load_from_checkpoint("/app/neurips2023-benchmarks/analysis/regression/results/wind_resnet50_512_data_augmentation/version_9/checkpoints/model_epoch=32.ckpt")

    # Callback for model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath= logger.save_dir + '/' + logger.name + '/version_%d/checkpoints/' % logger.version,
        filename='model_{epoch}',
        monitor='validation_loss', 
        verbose=True,
        every_n_epochs=1,
        save_top_k = 5
        )

    # Setting up the lightning trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=hparam.device,
        max_epochs=config.MAX_EPOCHS,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback]
    )

    # Launch training session
    trainer.fit(regression_model, data_module)
    
    return "training finished"

# Main execution block
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", default='resnet18', help="Model architecture to use (default: resnet18)")
    parser.add_argument("--size", default=config.DOWNSAMPLE_SIZE, help="Image size for processing (default: 512)")
    parser.add_argument("--cropped", default=True, help="Whether to use cropped images (default: True)")
    parser.add_argument("--device", default=config.DEVICES, help="Device to use for training (default: 0)")
    parser.add_argument("--labels", default=config.LABELS, help="Labels to predict (default: pressure)")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels (default: 3)")
    parser.add_argument("--force_channels", action="store_true", help="Force using specified in_channels instead of auto-detecting")
    parser.add_argument("--image_dirs", action="append", help="List of image directories (can specify multiple)")
    parser.add_argument("--metadata_dirs", action="append", help="List of metadata directories (can specify multiple)")
    parser.add_argument("--metadata_jsons", action="append", help="List of metadata JSON files (can specify multiple)")
    args = parser.parse_args()

    print(train(args))
