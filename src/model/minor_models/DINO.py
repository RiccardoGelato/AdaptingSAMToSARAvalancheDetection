from unicodedata import normalize
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
# Add the directory containing the sam module to the Python path
sys.path.append(os.path.abspath("../"))
from sam.build_sam import sam_model_registry
from sam.modeling.common import LayerNorm2d
from typing import List
from PIL import Image
import copy


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, image_size=1024):
        flip_and_color_jitter = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            v2.RandomGrayscale(p=0.2),
        ])
    
        # first global crop
        self.global_transfo1 = v2.Compose([
            v2.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            v2.GaussianBlur(kernel_size=3, sigma=0.01),
        ])
        # second global crop
        self.global_transfo2 = v2.Compose([
            v2.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            v2.GaussianBlur(kernel_size=3, sigma=0.1),
            #v2.Solarization(0.2),
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = v2.Compose([
            v2.RandomResizedCrop(int(96/(244/image_size)), scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            v2.GaussianBlur(kernel_size=3, sigma=0.01),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        tensor_transformation = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        return tensor_transformation(image), crops

local_crops_number = 0

data_transform = DataAugmentationDINO(
    global_crops_scale=(0.4, 1.0), 
    local_crops_scale=(0.05, 0.4), 
    local_crops_number=local_crops_number
)


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
import torch.nn.functional as F

import dataprocessing.slope as slope
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
        pixel_mean: List[float] = [0.5183216317041373, 0.5139202875277856, 0.1656409264909524],
        pixel_std: List[float] = [0.2547065818875149, 0.2571921631489484, 0.06727353387015675],
        normalize = True
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.normalize = normalize
    
    def forward(self, pixel_values):

        if self.normalize:
            pixel_values = self.preprocess(pixel_values)

        image_embeddings = self.image_encoder(pixel_values)  # (B, 256, 64, 64)
        
        return image_embeddings
    
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


class selfSupSamModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate=1e-3, startOver=True, normalize = True, use_scheduler=True, ema_momentum=0.96, center_momentum=0.9, tps=0.1, tpt=0.04 ):
        super(selfSupSamModel, self).__init__()

        model = sam_model_registry["vit_b"](os.path.join(SAM_CHECKPOINT, "/sam_vit_b_01ec64.pth"), adapt = True, selfsup=False) #selfsup false becuse we don't want added masking in this case

        
        self.model_student = selfSuperSAM(model.image_encoder, normalize=normalize)
        self.model_teacher = selfSuperSAM(copy.deepcopy(model.image_encoder), normalize=normalize)  # Teacher model for consistency loss

        # Freeze specific weights
        for name, param in self.model_student.named_parameters():
            if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
                if "Adapter" not in name and "patch_embed" not in name:
                    param.requires_grad = False 
        # Disable gradient updates for the teacher
        for param in self.model_teacher.parameters():
            param.requires_grad = False 

        self.learning_rate = learning_rate
        self.seg_loss = nn.MSELoss()
        self.test_outputs = []  # Initialize the test_outputs attribute
        self.use_scheduler = use_scheduler
        self.ema_momentum = ema_momentum
        self.center_momentum = center_momentum
        self.tps = tps
        self.tpt = tpt

        # Initialize center C as a buffer, K=256 for the embedding dimention of SAM
        K = 256 
        #self.register_buffer("center", torch.zeros(K))
        self.register_buffer("center", torch.zeros(256, 64, 64))

    def forward(self, x):
        # Normally, we assume x is a list of two augmented (global) views.
        with torch.no_grad():
            t1 = self.model_teacher(x[0])
            t2 = self.model_teacher(x[1])
        s1 = self.model_student(x[0])
        s2 = self.model_student(x[1])
        return s1, s2, t1, t2

    def training_step(self, batch, batch_idx):
        # Assume batch contains a key "views" which is a list of 2 global augmented views.
        #views = batch["views"]  # list of two tensors, each shape: (n, C, H, W)
        views = [batch["view1"], batch["view2"]]
        s1, s2, t1, t2 = self(views)
        loss = 0.5 * (self.dino_loss(t1, s2) + self.dino_loss(t2, s1))
        self.log('train_loss', loss, on_epoch=True, on_step=True)

        # Teacher update with exponential moving average of student parameters
        self._update_teacher()

        # Update center C using current teacher outputs
        new_center = torch.cat([t1, t2], dim=0).mean(dim=0)
        self.center = self.center * self.center_momentum + (1 - self.center_momentum) * new_center

        return loss

    def _update_teacher(self):
        # Updates teacher by mixing existing teacher params with student params.
        for teacher_param, student_param in zip(self.model_teacher.parameters(), self.model_student.parameters()):
            teacher_param.data = teacher_param.data * self.ema_momentum + student_param.data * (1 - self.ema_momentum)
    
    def dino_loss(self, t, s):
        """
        Computes cross-entropy loss H as described in the pseudocode.
          t: teacher logits (n-by-K)
          s: student logits (n-by-K)
        """
        # Stop gradient on teacher
        t = t.detach()
        # student: apply temperature scaling and softmax
        s = F.softmax(s / self.tps, dim=1)
        # teacher: center and sharpen with teacher temp (tpt)
        t = F.softmax((t - self.center) / self.tpt, dim=1)
        # Cross-entropy between teacher and student (averaged over batch)
        return - (t * torch.log(s + 1e-6)).sum(dim=1).mean()

    
    def validation_step(self, batch, batch_idx):
        # Assume batch contains a key "views" which is a list of 2 global augmented views.
        #views = batch["views"]  # list of two tensors, each shape: (n, C, H, W)
        views = [batch["view1"], batch["view2"]]
        s1, s2, t1, t2 = self(views)
        loss = 0.5 * (self.dino_loss(t1, s2) + self.dino_loss(t2, s1))
        self.log('val_loss', loss, on_epoch=True, on_step=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        # Assume batch contains a key "views" which is a list of 2 global augmented views.
        #views = batch["views"]  # list of two tensors, each shape: (n, C, H, W)
        views = [batch["view1"], batch["view2"]]
        s1, s2, t1, t2 = self(views)
        test_loss = 0.5 * (self.dino_loss(t1, s2) + self.dino_loss(t2, s1))


        # Store results
        self.log('test_loss', test_loss)
        self.test_outputs.append({
            'test_loss': test_loss,
        })
        return {
            'test_loss': test_loss,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  # Metric to monitor for ReduceLROnPlateau
        }
    
    

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
    #label = np.array(item["label"], np.float32)

   # print("image shape:", image.shape)

    # prepare image and prompt for the model
    inputs = self.processor(image, do_normalize = self.normalize, return_tensors="pt", do_rescale = self.rescale)
    #ground_truth_mask = self.processor(label ,do_normalize = self.normalize, return_tensors="pt", do_resize = False, do_pad = False, do_rescale = self.rescale)
    # remove batch dimension which the processor adds by default
    inputs = {k:v for k,v in inputs.items()}

    #print("inputs['pixel_values'] shape:", inputs["pixel_values"].shape)

    #if self.augment:
    # Apply DINO augmentations; this returns a tuple: (toTensor(image), list_of_augmented_crops)
    image_tensor, crops = data_transform(inputs["pixel_values"])

    #print("image_tensor shape:", image_tensor.shape)
    #print("crops shape:", crops.shape)

    # You can store global view and local crops separately.
    
    inputs = {
        "pixel_values": image_tensor,  # the normalized version of the original image
        #"views": crops,                # list of augmented views (e.g., 2 global + several local crops)
        "view1": crops[0],  # first global view
        "view2": crops[1],  # second global view
    }
    

    return inputs