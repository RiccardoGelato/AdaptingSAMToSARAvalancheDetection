import torch
import pytorch_lightning as pl
from datasets import Dataset, load_from_disk
from torchvision.transforms import v2
import torch.nn as nn
import monai
from segment_anything.build_sam import sam_model_registry
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import List
from dataprocessing.rcsHandlingFunctions import _rescale
from model.inputTypes import InputTypes

class selfSuperSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        target_image_encoder,
        pixel_mean: List[float] = [0.5183216317041373, 0.5139202875277856, 0.1656409264909524], #normalMean
        pixel_std: List[float] = [0.2547065818875149, 0.2571921631489484, 0.06727353387015675],#normalStd
        pixel_mean_final: List[float] = [0.4224749803543091, 0.3967047929763794, 0.5431008338928223, 0.5183878540992737, 0.16572988033294678, 0.24865785241127014], #normalMean
        pixel_std_final: List[float] = [0.23271377384662628, 0.2313859462738037, 0.2603246569633484, 0.2645578682422638, 0.06731900572776794, 0.15417179465293884], #normalStd
        pixel_mean_VV: List[float] = [0.5431008338928223, 0.5183878540992737, 0.16572988033294678], #VVMean
        pixel_std_VV: List[float] = [0.2603246569633484, 0.2645578682422638, 0.06731900572776794], #VVStd
        pixel_mean_VH: List[float] = [0.4224749803543091, 0.3967047929763794, 0.24865785241127014], #VHMean
        pixel_std_VH: List[float] = [0.23271377384662628, 0.2313859462738037, 0.15417179465293884], #VHStd
        input_type: str = InputTypes.Normal,  # "all", "VV", "VH", "normal"
        target_input_type: str = InputTypes.Normal,  # "all", "VV", "VH", "normal"
        normalize = True
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.target_image_encoder = target_image_encoder
        if input_type == InputTypes.All:
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean_final).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std_final).view(-1, 1, 1), False)
        elif input_type == InputTypes.VV:
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean_VV).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std_VV).view(-1, 1, 1), False)
        elif input_type == InputTypes.VH:
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean_VH).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std_VH).view(-1, 1, 1), False)
        else:
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        if target_input_type == InputTypes.All:
            self.register_buffer("target_pixel_mean", torch.Tensor(pixel_mean_final).view(-1, 1, 1), False)
            self.register_buffer("target_pixel_std", torch.Tensor(pixel_std_final).view(-1, 1, 1), False)
        elif target_input_type == InputTypes.VV:
            self.register_buffer("target_pixel_mean", torch.Tensor(pixel_mean_VV).view(-1, 1, 1), False)
            self.register_buffer("target_pixel_std", torch.Tensor(pixel_std_VV).view(-1, 1, 1), False)
        elif target_input_type == InputTypes.VH:
            self.register_buffer("target_pixel_mean", torch.Tensor(pixel_mean_VH).view(-1, 1, 1), False)
            self.register_buffer("target_pixel_std", torch.Tensor(pixel_std_VH).view(-1, 1, 1), False)
        else:
            self.register_buffer("target_pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("target_pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.normalize = normalize
    
    def forward(self, pixel_values, target_pixel_values):

        if self.normalize:
            pixel_values = self.preprocess(pixel_values)

        image_embeddings = self.image_encoder(pixel_values)  # (B, 256, 64, 64)

        with torch.no_grad():
            target_pixel_values = self.preprocess_target(target_pixel_values)
            target_image_embeddings = self.target_image_encoder(target_pixel_values)  # (B, 256, 64, 64)

        return [image_embeddings, target_image_embeddings]
    
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
    
    def preprocess_target(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.target_pixel_mean) / self.target_pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = nn.functional.pad(x, (0, padw, 0, padh))
        return x


class selfSupSamModel(pl.LightningModule):
    def __init__(self, image_encoder, target_image_encoder, learning_rate=1e-3, normalize=True, input_type=InputTypes.All, target_input_type=InputTypes.Normal):
        super(selfSupSamModel, self).__init__()
        
        self.model = selfSuperSAM(
            image_encoder=image_encoder,
            target_image_encoder=target_image_encoder,
            normalize=normalize, 
            input_type=input_type,
            target_input_type=target_input_type
        )  

        # Freeze specific weights
        for name, param in self.model.named_parameters():
            if name.startswith("image_encoder"):
                if "Adapter" not in name:
                    param.requires_grad = False 
            else:
                param.requires_grad = False

        self.learning_rate = learning_rate
        self.seg_loss = nn.MSELoss()
        self.test_outputs = []  # Initialize the test_outputs attribute

    def forward(self, pixel_values, target_pixel_values):
        return self.model(pixel_values=pixel_values, target_pixel_values=target_pixel_values)

    def training_step(self, batch, batch_idx):
        predicted_embeddings = self(pixel_values=batch["pixel_values"], target_pixel_values=batch["target_pixel_values"])
        loss = self.seg_loss(predicted_embeddings[0], predicted_embeddings[1])
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        predicted_embeddings = self(pixel_values=batch["pixel_values"], target_pixel_values=batch["target_pixel_values"])
        val_loss = self.seg_loss(predicted_embeddings[0], predicted_embeddings[1])
        self.log('val_loss', val_loss, on_epoch=True, on_step=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        predicted_embeddings = self(pixel_values=batch["pixel_values"], target_pixel_values=batch["target_pixel_values"])
        test_loss = self.seg_loss(predicted_embeddings[0], predicted_embeddings[1])


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
        #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        #return optimizer
    
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
    

    
class RandomFlip:
    def __init__(self, p=0.66):
        self.p = p

    def __call__(self, inputs):
        img = inputs["pixel_values"]
        target_image = inputs["target_pixel_values"]

        if np.random.random() < self.p:
            if np.random.choice([True, False]):
                # Apply horizontal flip
                img = v2.functional.hflip(img)
                target_image = v2.functional.hflip(target_image)
            else:
                # Apply vertical flip
                img = v2.functional.vflip(img)
                target_image = v2.functional.vflip(target_image)

        inputs["pixel_values"] = img
        inputs["target_pixel_values"] = target_image
        return inputs


class RandomNoise:
    def __init__(self, noise_factor=0.01):
        self.noise_factor = noise_factor

    def __call__(self, inputs):
        img = inputs["pixel_values"]

        inputs["pixel_values"] = v2.GaussianNoise(clip=False, sigma= self.noise_factor)(img)

        return inputs
    
class RandomAffine:
    def __init__(self, max_translation=50):
        """
        Args:
            max_translation (int): Maximum translation (in px) in both x and y directions.
        """
        self.max_translation = max_translation

    def __call__(self, inputs):

        img = inputs["pixel_values"]
        target = inputs["target_pixel_values"]

        # Choose random rotation angle
        angle = np.random.uniform(0, 360)
        # Choose random translation offsets (dx, dy)
        tx = np.random.randint(-self.max_translation, self.max_translation)
        ty = np.random.randint(-self.max_translation, self.max_translation)
        # Use affine transform with angle rotation, scale=1, no shear.
        img = v2.functional.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=0.0)
        target = v2.functional.affine(target, angle=angle, translate=(tx, ty), scale=1.0, shear=0.0)

        inputs["pixel_values"] = img
        inputs["target_pixel_values"] = target
        return inputs

class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor, do_normalize = False, do_rescale = False, augment = True, type = InputTypes.All, target_type = InputTypes.All, target_size = 1024):
    self.dataset = dataset
    self.processor = processor
    self.normalize = do_normalize
    self.rescale = do_rescale
    self.augment = augment
    self.type = type
    self.target_type = target_type
    self.target_size = target_size


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

    if self.type == InputTypes.VV:
        image = np.stack([VV0, VV1, dem], axis=1)
    elif self.type == InputTypes.VH:
        image = np.stack([VH0, VH1, slope], axis=1)
    elif self.type == InputTypes.Normal:
        a = _rescale(VH1 - VH0, 0, .25)

        b = _rescale(VV1 - VV0, 0, .25)

        w = _rescale(a - b, 0, 1)

        r = w*VH0 + (1 - w)*VV0

        g = w*VH1 + (1 - w)*VV1

        image = np.stack([r, g, dem], axis=1)
    else:
        # Combine all channels to create image
        image = np.stack([VH0, VH1, VV0, VV1, dem, slope], axis=1)

    if self.target_type == InputTypes.VV:
        target_image = np.stack([VV0, VV1, dem], axis=1)
    elif self.target_type == InputTypes.VH:
        target_image = np.stack([VH0, VH1, slope], axis=1)
    elif self.target_type == InputTypes.Normal:
        a = _rescale(VH1 - VH0, 0, .25)
        b = _rescale(VV1 - VV0, 0, .25)
        w = _rescale(a - b, 0, 1)
        r = w*VH0 + (1 - w)*VV0
        g = w*VH1 + (1 - w)*VV1
        target_image = np.stack([r, g, dem], axis=1)
    else:
        # Combine all channels to create target image
        target_image = np.stack([VH0, VH1, VV0, VV1, dem, slope], axis=1)

    image_tensor = torch.from_numpy(image).float()  # Convert to tensor and change to (C, H, W)
    target_image_tensor = torch.from_numpy(target_image).float()  # Convert to tensor and change to (C, H, W)
    


    inputs = {
            "pixel_values": image_tensor,
            "target_pixel_values": target_image_tensor
        }
    

    if self.augment:
        inputs = RandomAffine()(inputs)
        inputs = RandomFlip()(inputs)
        inputs = RandomNoise()(inputs)

    inputs["pixel_values"] = torch.nn.functional.interpolate(inputs["pixel_values"], size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
    inputs["target_pixel_values"] = torch.nn.functional.interpolate(inputs["target_pixel_values"], size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)

    return inputs