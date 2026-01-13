from datasets import load_from_disk
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from transformers import SamProcessor
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.adapterModel import LitSamModel
from model.samDataset import SAMDataset3
import warnings
import random
import numpy as np
import pickle
from model.inputTypes import InputTypes

# Set a global seed for reproducibility
seed = 46

# Python's built-in random module
random.seed(seed)

# Numpy
np.random.seed(seed)

# PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# For PyTorch Lightning
pl.seed_everything(seed, workers=True)

# Suppress the specific user warning about invalid arguments
warnings.filterwarnings(
    "ignore",
    message=".*The following named arguments are not valid for `SamImageProcessor.preprocess`.*"
)

import yaml
import os
from pathlib import Path

# 1. Get the path of the script
current_file = Path(__file__).resolve() # src/training/your_script.py

# 2. Go up one level to 'src', then into 'config'
config_path = current_file.parent.parent / "config" / "config_general.yaml"

# 3. Load the YAML
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 4. Resolve the root of the project (one level above 'src')
# This ensures that "./data" in the YAML is interpreted relative to the Project_Root
PROJECT_ROOT = current_file.parent.parent.parent
os.chdir(PROJECT_ROOT) 

# Extract paths from YAML
DATA_DIR = config['paths']['data']
CHECKPOINT_DIR = config['paths']['checkpoints']
SAM_CHECKPOINT = config['paths']['sam_checkpoint']

def reduce_precision():
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

reduce_precision()

from torch.utils.data._utils.collate import default_collate

def custom_collate(batch):
    collated = {}
    for key in batch[0]:
        if key == "input_boxes":
            # Leave input_boxes as a list so that each sample can have different numbers of boxes.
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = default_collate([item[key] for item in batch])
    return collated

#train_dataset = load_from_disk('/home/gelato/Avalanche-Segmentation-with-Sam/code/dataprocessing/datasetTrainDEMSmall')
#val_dataset = load_from_disk('/home/gelato/Avalanche-Segmentation-with-Sam/code/dataprocessing/datasetValDEMSmall')
train_dataset = load_from_disk(os.path.join(DATA_DIR, 'datasetTrainFinal'))
val_dataset = load_from_disk(os.path.join(DATA_DIR, 'datasetValFinal'))

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

with open("val_indices.pkl", "rb") as f:
    val_indices = pickle.load(f)

with open("train_indices.pkl", "rb") as f:
    train_indices = pickle.load(f)

# Used to train only on samples with the full mask
all_indices = set(range(len(train_dataset)))
exclude_indices = set(train_indices)
train_indices = list(all_indices - exclude_indices)
all_indices = set(range(len(val_dataset)))
exclude_indices = set(val_indices)
val_indices = list(all_indices - exclude_indices)

# Filter the dataset to remove samples with a bbox of [0,0,512,512]
train_dataset = train_dataset.select(train_indices)

val_dataset = val_dataset.select(val_indices)

# Create an instance of the SAMDataset
train_dataset_sam = SAMDataset3(dataset=train_dataset, processor=processor, augment=True, type = InputTypes.VH)

# Create a DataLoader instance for the training dataset
train_dataloader = DataLoader(train_dataset_sam, batch_size=5, shuffle=True, drop_last=True, num_workers=8, collate_fn=custom_collate)

# Create an instance of the SAMDataset
val_dataset_sam = SAMDataset3(dataset=val_dataset, processor=processor, augment=False, type = InputTypes.VH)

# Create a DataLoader instance for the validation dataset
val_dataloader = DataLoader(val_dataset_sam, batch_size=5, shuffle=False, drop_last=True, num_workers=8, collate_fn=custom_collate)

# Define the model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_iou',  # Metric to monitor
    dirpath=CHECKPOINT_DIR,  # Directory to save checkpoints
    filename='sam-adapter-complementaryVH2-{epoch:02d}-{val_loss:.3f}-{val_iou:.3f}',  # Filename template
    save_top_k=3,  # Save the top 3 models
    mode='max'  # Mode can be 'min' or 'max'. Use 'min' for loss and 'max' for accuracy
)

# Define the early stopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_iou',  # Metric to monitor
    patience=30,  # Number of epochs with no improvement after which training will be stopped
    verbose=True,
    mode='max'  # Mode can be 'min' or 'max'. Use 'min' for loss and 'max' for accuracy
)

lr_monitor = LearningRateMonitor(logging_interval='step')

sam_checkpoint = "/home/gelato/Avalanche-Segmentation-with-Sam/code/training/checkpoints/firstencoder/sam-adapters-2-epoch=70-val_loss=0.226-val_iou=0.597.ckpt"

# Initialize the model
modelComplementary = LitSamModel.load_from_checkpoint(sam_checkpoint, model_name="vit_b", normalize=True, learning_rate=1e-3, input_type=InputTypes.VH)

model = LitSamModel(model_name="vit_b", normalize=True, learning_rate=1e-4, adapt_patch_embed=False, decoder = modelComplementary.model.mask_decoder, input_type=InputTypes.VH)

#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name)

import neptune
from pytorch_lightning.loggers import NeptuneLogger

# Extract neptune config
neptune_conf = config.get('neptune', {})

# Initialize the Neptune logger
neptune_logger = NeptuneLogger(
    api_token=neptune_conf.get("api_token") or os.getenv("NEPTUNE_API_TOKEN"),
    project=neptune_conf.get("project"),
    log_model_checkpoints=neptune_conf.get("log_model_checkpoints", True)
)

# Optional: Add tags from config
if "tags" in neptune_conf:
    neptune_logger.run["sys/tags"].add(neptune_conf["tags"])


# Initialize the trainer with logging and checkpointing
trainer = pl.Trainer(
    max_epochs=1000,  # Maximum number of epochs
    accelerator='gpu',  # Use GPU if available
    devices=1,  # Use GPUs if available
    callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],  # Add the callbacks
    #logger=neptune_logger  # Enable logging
)


# Train the model
trainer.fit(model, train_dataloader, val_dataloader)

