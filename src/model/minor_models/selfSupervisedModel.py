import torch
import pytorch_lightning as pl
from datasets import Dataset, load_from_disk
from torchvision.transforms import v2
import torch.nn as nn
import monai
from sam.build_sam import sam_model_registry
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import sys
import os
from sam.build_sam import sam_model_registry
from sam.modeling.common import LayerNorm2d
from typing import List
import sam.modeling.PMD_features as pmd

import dataprocessing.slope as slope
from dataprocessing.rcsHandlingFunctions import _rescale
from model.imageAugmentations import RandomAffine
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

class selfSuperSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        #pixel_mean: List[float] = [0.5109968781471252, 0.5048336982727051, 0.16753898561000824],
        #pixel_std: List[float] = [0.25895485281944275, 0.2620321214199066, 0.0687190517783165],
        #pixel_mean: List[float] = [0.5183216317041373, 0.5139202875277856, 0.1656409264909524],
        #pixel_std: List[float] = [0.2547065818875149, 0.2571921631489484, 0.06727353387015675],
        pixel_mean: List[float] = [0.4224749803543091, 0.3967047929763794, 0.5431008338928223, 0.5183878540992737, 0.16572988033294678, 0.24865785241127014], #normalMean
        pixel_std: List[float] = [0.23271377384662628, 0.2313859462738037, 0.2603246569633484, 0.2645578682422638, 0.06731900572776794, 0.15417179465293884], #normalStd
        normalize = True
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_decoder = LightDecoder(input_dim=256, output_dim=6) #TransformerDecoder(input_dim=256, output_dim=3)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.normalize = normalize
    
    def forward(self, pixel_values):

        if self.normalize:
            pixel_values = self.preprocess(pixel_values)

        image_embeddings = self.image_encoder(pixel_values)  # (B, 256, 64, 64)

        output = self.image_decoder(image_embeddings)
        
        return output
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = nn.functional.pad(x, (0, padw, 0, padh))
        return x
    

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, num_layers=2, num_heads=8, hidden_dim=2048, output_dim=1, activation: nn.Module = nn.GELU,):
        super(TransformerDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(input_dim, input_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(input_dim // 4),
            activation(),
            nn.ConvTranspose2d(input_dim // 4, input_dim // 8, kernel_size=2, stride=2),
            activation(),
            LayerNorm2d(input_dim // 8),
            activation(),
            nn.ConvTranspose2d(input_dim // 8, output_dim, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

class LightDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=1, activation: nn.Module = nn.GELU,):
        super(LightDecoder, self).__init__()
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(input_dim, input_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(input_dim // 4),
            activation(),
            nn.ConvTranspose2d(input_dim // 4, input_dim // 8, kernel_size=2, stride=2),
            activation(),
            LayerNorm2d(input_dim // 8),
            activation(),
            nn.ConvTranspose2d(input_dim // 8, output_dim, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, image_embeddings):
        # x: (T, B, C)
        # memory: (S, B, C)
        x = self.output_layer(image_embeddings)
        return x


class selfSupSamModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate=1e-3, normalize = True, use_scheduler=True, adapt_patch_embed=False):
        super(selfSupSamModel, self).__init__()

        model = sam_model_registry["vit_b"](os.path.join(SAM_CHECKPOINT, "/sam_vit_b_01ec64.pth"), adapt = True, selfsup=True, adapt_patch_embed=adapt_patch_embed)
        
        self.model = selfSuperSAM(model.image_encoder, normalize = normalize)  

        # Freeze specific weights
        for name, param in self.model.named_parameters():
            if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
                if "Adapter" not in name and "patch_embed" not in name:
                    param.requires_grad = False 

        self.learning_rate = learning_rate
        self.seg_loss = nn.MSELoss()
        self.test_outputs = []  # Initialize the test_outputs attribute
        self.use_scheduler = use_scheduler

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)

    def training_step(self, batch, batch_idx):
        predicted_masks = self(pixel_values=batch["pixel_values"])
        predicted_masks = torch.nn.functional.interpolate(predicted_masks,
                size=(512, 512),
                mode='bilinear',
                align_corners=False)
        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
        loss = self.seg_loss(predicted_masks, ground_truth_masks)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        predicted_masks = self(pixel_values=batch["pixel_values"])
        predicted_masks = torch.nn.functional.interpolate(predicted_masks,
                size=(512, 512),
                mode='bilinear',
                align_corners=False)
        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
        val_loss = self.seg_loss(predicted_masks, ground_truth_masks)
        self.log('val_loss', val_loss, on_epoch=True, on_step=True)
        self.maybe_log_preds(batch_idx, predicted_masks, ground_truth_masks)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        predicted_masks = self(pixel_values=batch["pixel_values"])
        predicted_masks = torch.nn.functional.interpolate(predicted_masks,
                size=(512, 512),
                mode='bilinear',
                align_corners=False)
        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
        test_loss = self.seg_loss(predicted_masks, ground_truth_masks)

        predicted_masks = predicted_masks.squeeze(1)


        # Store results
        self.log('test_loss', test_loss)
        self.test_outputs.append({
            'test_loss': test_loss,
            'ground_truth': ground_truth_masks.cpu().numpy(),
            'predicted_masks': predicted_masks.cpu().numpy()
        })
        return {
            'test_loss': test_loss,
            'ground_truth': ground_truth_masks.cpu().numpy(),
            'predicted_masks': predicted_masks.cpu().numpy()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  # Metric to monitor for ReduceLROnPlateau
        }
    
    def maybe_log_preds(self, batch_idx, prediction, ground_truth):
        # Check for a logger and that it has the log_tensor_img attribute
        if not hasattr(self, "logger") or self.logger is None or not hasattr(self.logger, "log_tensor_img"):
            return
        every = 50
        if batch_idx == 0 and self.current_epoch % every == 0:
            
            img = prediction[0] # shape [C,H,W]
            title = f'selfsupervised_pred@ep{self.current_epoch}'
            # Neptune expects an HxWxC, so permute for logging:
            img_np = img.permute(1,2,0)  # shape [H,W,3]
            self.logger.log_tensor_img(img_np, title)

            # Log the ground truth mask:
            gt_img = ground_truth[0]  # shape [C, H, W]
            gt_title = f'selfsupervised_gt@ep{self.current_epoch}'
            gt_img_np = gt_img.permute(1, 2, 0)
            self.logger.log_tensor_img(gt_img_np, gt_title)
    
    

class RandomRotation90:
    def __call__(self, inputs):
        img = inputs["pixel_values"]
        mask = inputs["ground_truth_mask"]
        angle = np.random.choice([0, 90, 180, 270])
        img = v2.functional.rotate(img, angle)
        mask = v2.functional.rotate(mask, angle)
        inputs["pixel_values"] = img
        inputs["ground_truth_mask"] = mask
        return inputs

    
class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, inputs):
        img = inputs["pixel_values"]
        mask = inputs["ground_truth_mask"]

        if np.random.random() < self.p:
            if np.random.choice([True, False]):
                # Apply horizontal flip
                img = v2.functional.hflip(img)
                mask = v2.functional.hflip(mask)
            else:
                # Apply vertical flip
                img = v2.functional.vflip(img)
                mask = v2.functional.vflip(mask)

        inputs["pixel_values"] = img
        inputs["ground_truth_mask"] = mask
        return inputs

class RandomMasking:
    def __init__(self, max_size=64, mask_value=0, max_masks=100):
        self.max_size = max_size
        self.mask_value = mask_value
        self.max_masks = max_masks

    def __call__(self, inputs):
        img = inputs["pixel_values"]

        # Get image dimensions
        _,_, h, w = img.shape

        # Randomly choose the number of masks to apply
        num_masks = np.random.randint(0, self.max_masks + 1)

        # Randomly choose the size of the mask
        mask_size = np.random.randint(1, self.max_size + 1)

        for _ in range(num_masks):

            #APPLY INDIPEMDENTLY TO EACH LAYER
            #for layer in range(3):
            ## Randomly choose the top-left corner of the mask
            #    top = np.random.randint(0, h - mask_size)
            #    left = np.random.randint(0, w - mask_size)
            #
            #    # Apply the mask to the image
            #    img[:, layer, top:top + mask_size, left:left + mask_size] = self.mask_value

            #APPLY TO ALL LAYERS
            top = np.random.randint(0, h - mask_size)
            left = np.random.randint(0, w - mask_size)
            img[:, :, top:top + mask_size, left:left + mask_size] = self.mask_value

        inputs["pixel_values"] = img
        return inputs
    
class GridMasking:
    def __init__(self, grid_size=32, mask_value=0, mask_fraction=0.3):
        self.grid_size = grid_size
        self.mask_value = mask_value
        self.mask_fraction = mask_fraction

    def __call__(self, inputs):
        img = inputs["pixel_values"]

        # Get image dimensions
        _, _, h, w = img.shape

        # For example, with a standard deviation of 2:
        random_value = int(np.random.normal(loc=self.grid_size, scale=2))

        # Calculate the size of each grid cell
        cell_h = h // random_value#self.grid_size
        cell_w = w // random_value#self.grid_size

        # Create a list of all grid cells
        #cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        cells = [(i, j) for i in range(random_value) for j in range(random_value)]

        # Randomly select a subset of cells to mask
        num_cells_to_mask = int(len(cells) * self.mask_fraction)
        cells_to_mask = np.random.choice(len(cells), num_cells_to_mask, replace=False)

        for idx in cells_to_mask:
            i, j = cells[idx]
            top = i * cell_h
            left = j * cell_w
            img[:, :, top:top + cell_h, left:left + cell_w] = self.mask_value

        inputs["pixel_values"] = img
        return inputs
    
class JigsawPuzzle:
    def __init__(self, grid_size=3):
        self.grid_size = grid_size

    def __call__(self, inputs):
        img = inputs["pixel_values"]

        # Get image dimensions
        _, h, w = img.shape

        # Calculate the size of each patch
        patch_h = h // self.grid_size
        patch_w = w // self.grid_size

        # Create a list of patches
        patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch = img[:, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
                patches.append(patch)

        # Shuffle the patches
        np.random.shuffle(patches)

        # Reassemble the image from the shuffled patches
        shuffled_img = torch.zeros_like(img)
        idx = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                shuffled_img[:, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w] = patches[idx]
                idx += 1

        inputs["pixel_values"] = shuffled_img
        return inputs
    
class RandomNoise:
    def __init__(self, noise_factor=0.01):
        self.noise_factor = noise_factor

    def __call__(self, inputs):
        img = inputs["pixel_values"]

        inputs["pixel_values"] = v2.GaussianNoise(clip=False, sigma= self.noise_factor)(img)

        return inputs

class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor, do_normalize = False, do_rescale = False, augment = True):
    self.dataset = dataset
    self.processor = processor
    self.normalize = do_normalize
    self.rescale = do_rescale
    self.augment = augment


  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = np.array(item["image"], np.float32)
    label = np.array(item["label"], np.float32)

    # prepare image and prompt for the model
    inputs = self.processor(image, do_normalize = self.normalize, return_tensors="pt", do_rescale = self.rescale)
    ground_truth_mask = self.processor(label ,do_normalize = self.normalize, return_tensors="pt", do_resize = False, do_pad = False, do_rescale = self.rescale)
    # remove batch dimension which the processor adds by default
    inputs = {k:v for k,v in inputs.items()}
    
    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask["pixel_values"]
    if self.augment:
        inputs = RandomRotation90()(inputs)
        inputs = RandomFlip()(inputs)
    #inputs = RandomMasking()(inputs)
    #inputs = GridMasking()(inputs)
    

    return inputs
  

class SAMDataset3(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor, do_normalize = False, do_rescale = False, augment = True, target_size=1024, type = "all", test = False):
    self.dataset = dataset
    self.processor = processor
    self.normalize = do_normalize
    self.rescale = do_rescale
    self.augment = augment
    self.target_size = target_size
    self.type = type
    self.test = test

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    VH0 = np.array(item["VH0"])
    VH1 = np.array(item["VH1"])
    VV0 = np.array(item["VV0"])
    VV1 = np.array(item["VV1"])
    dem = np.array(item["dem"])
    slope = np.array(item["slope"])

    if self.type == "VV":
        image = np.stack([VV0, VV1, dem], axis=1)
    elif self.type == "VH":
        image = np.stack([VH0, VH1, dem], axis=1)
    elif self.type == "combine":
        a = _rescale(VH1 - VH0, 0, .25)

        b = _rescale(VV1 - VV0, 0, .25)

        w = _rescale(a - b, 0, 1)

        r = w*VH0 + (1 - w)*VV0

        g = w*VH1 + (1 - w)*VV1

        image = np.stack([r, g, dem], axis=1)
    else:
        # Combine all channels to create image
        image = np.stack([VH0, VH1, VV0, VV1, dem, slope], axis=1)

    image_tensor = torch.from_numpy(image).float()  # Convert to tensor and change to (C, H, W)

    inputs = {
            "pixel_values": image_tensor,
            "ground_truth_mask": image_tensor,
        }
    
    if self.augment:
        inputs = RandomAffine()(inputs)
        inputs = RandomFlip()(inputs)
        inputs = RandomNoise()(inputs)

    inputs["pixel_values"] = torch.nn.functional.interpolate(inputs["pixel_values"], size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)

    return inputs