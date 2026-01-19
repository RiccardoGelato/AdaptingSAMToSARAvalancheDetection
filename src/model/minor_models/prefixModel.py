import torch
import pytorch_lightning as pl
import torch.nn as nn
import monai
from sam.build_sam import sam_model_registry
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sam.build_sam import sam_model_registry
from sam.modeling.common import LayerNorm2d
from typing import List
from datasets import Dataset
from torchvision.transforms import v2
import lora.loraModules as lora
from model.convNet import Prefix, PrefixSmall
from model.baseModel import BaseSamModel
from model.imageAugmentations import RandomRotation90, RandomFlip, RandomMasking, RandomNoise, RandomAffine, process_meteo
from dataprocessing.rcsHandlingFunctions import _rescale
from dataprocessing.creationOfDataframe import create_bounding_boxes
from model.inputTypes import InputTypes

class PrefixSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        prefix = PrefixSmall(6),     #Prefix(6),
        normalize = True,
        pixel_mean: List[float] = [x / 255.0 for x in [123.675, 116.28, 103.53]],  # imagenetMean
        pixel_std: List[float] = [x / 255.0 for x in [58.395, 57.12, 57.375]],    # imagenetStd
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.prefix = prefix
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.normalize = normalize
    
    def forward(self, pixel_values, input_boxes, slopes=None, meteo_data=None):

        pixel_values = self.prefix(pixel_values)  # (B, 3, 512, 512)

        outputs = []
        batch_size = pixel_values.shape[0]
        
        # Optionally, apply normalization once for each image.
        if self.normalize:
            pixel_values = self.preprocess(pixel_values)

        image_embeddings = self.image_encoder(pixel_values)  # (B, 256, 64, 64)

        
        for i in range(batch_size):
            if len(input_boxes[0].shape) == 1:
                # If only one box per image, add a dimension
                input_boxes = input_boxes.unsqueeze(1) #from (B, 4) to (B, 1, 4)
            bounding_boxes = input_boxes[i]
            number_of_boxes = bounding_boxes.shape[0]
            embedding_of_image = image_embeddings[i].unsqueeze(0)  # (1, 256, 64, 64)
            for box_index in range(number_of_boxes):
                # Process prompt encoder for this one image (with no gradient)
                with torch.no_grad():
                    # Assume input_boxes[i] is the prompt boxes for the i-th image
                    boxes = bounding_boxes[box_index].unsqueeze(0)  # (1, 4)
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points=None,
                        boxes=boxes,
                        masks=None,
                    )
            
                low_res_masks, _ = self.mask_decoder(
                    image_embeddings=embedding_of_image,         # (1, 256, 64, 64)
                    image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64) or broadcastable
                    sparse_prompt_embeddings=sparse_embeddings,  # (1, 2, 256) 
                    dense_prompt_embeddings=dense_embeddings,    # (1, 256, 64, 64)
                    multimask_output=False,
                )
                outputs.append(low_res_masks)
        
        low_res_masks = torch.cat(outputs, dim=0)
        
        return low_res_masks
    
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

class LitSamModel(BaseSamModel):
    def _build_model(self, image_encoder, mask_decoder, prompt_encoder, normalize=True, adapt_patch_embed=False, input_type: InputTypes=InputTypes.Normal):
        return PrefixSAM(
            image_encoder=image_encoder,
            mask_decoder=mask_decoder,
            prompt_encoder=prompt_encoder,
            normalize=normalize
        )

    def should_freeze(self, name, param):
        # Freeze image and prompt encoder except Adapters and LoRA layers
        if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
            if "Adapter" not in name and "lora" not in name:
                return True
        return False 


#class LitSamModel(pl.LightningModule):
#    def __init__(self, model_name, learning_rate=1e-3, adapt=False, normalize=True, encoder=None, prefix = Prefix(6)):
#        super(LitSamModel, self).__init__()
#
#        if model_name == "vit_l":
#            model = sam_model_registry["vit_l"]("sam_vit_l_0b3195.pth", adapt=adapt)
#        else:
#            model = sam_model_registry["vit_b"]("/home/gelato/Avalanche-Segmentation-with-Sam/code/model/sam_vit_b_01ec64.pth", adapt=adapt)
#        
#
#        if encoder is None:
#            self.model = PrefixSAM(model.image_encoder, model.mask_decoder, model.prompt_encoder, normalize=normalize, prefix=prefix)  
#        else:
#            self.model = PrefixSAM(encoder, model.mask_decoder, model.prompt_encoder, normalize=normalize, prefix=prefix) 
#        
#        #lora.replace_conv2d_with_lora(self.model, ["proj"], r=4, alpha=1.0)
#
#        # Freeze specific weights
#        for name, param in self.model.named_parameters():
#            if name.startswith("image_encoder") or name.startswith("prompt_encoder") or name.startswith("prefix"):
#                if "Adapter" not in name and "lora" not in name and "upconv" not in name and "lastConv" not in name and "final" not in name and "convMedium1" not in name:
#                    param.requires_grad = False
#
#        self.learning_rate = learning_rate
#        #Try DiceFocalLoss, FocalLoss, DiceCELoss
#        self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
#        #self.seg_loss = monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')
#        self.test_outputs = []  # Initialize the test_outputs attribute
#
#    def forward(self, pixel_values, input_boxes, slopes=None, meteo_data=None):
#        return self.model(pixel_values=pixel_values, input_boxes=input_boxes, slopes=None, meteo_data=None)
#
#    def training_step(self, batch, batch_idx):
#        predicted_masks = self(pixel_values=batch["pixel_values"], input_boxes=batch["input_boxes"], slopes=batch["slope"] if "slope" in batch else None, meteo_data=batch["meteo"] if "meteo" in batch else None)
#        predicted_masks = torch.nn.functional.interpolate(predicted_masks,
#                size=(512, 512),
#                mode='bilinear',
#                align_corners=False)
#        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
#        ground_truth_masks = self.compute_ground_truth_masks(ground_truth_masks, batch["input_boxes"]).to(self.device)
#        print(predicted_masks.shape, ground_truth_masks.unsqueeze(1).shape)
#        loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
#        iou = self.compute_iou(predicted_masks.squeeze(1), ground_truth_masks)
#        self.log('train_iou', iou, on_epoch=True, on_step=True)
#        self.log('train_loss', loss, on_epoch=True, on_step=True)
#        return loss
#    
#    def validation_step(self, batch, batch_idx):
#        predicted_masks = self(pixel_values=batch["pixel_values"], input_boxes=batch["input_boxes"], slopes=batch["slope"] if "slope" in batch else None)
#        predicted_masks = torch.nn.functional.interpolate(predicted_masks,
#                size=(512, 512),
#                mode='bilinear',
#                align_corners=False)
#        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
#
#        ground_truth_masks = self.compute_ground_truth_masks(ground_truth_masks, batch["input_boxes"]).to(self.device)
#
#        val_loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
#        iou = self.compute_iou(predicted_masks.squeeze(1), ground_truth_masks)
#        self.log('val_iou', iou, on_epoch=True, on_step=True)
#        self.log('val_loss', val_loss, on_epoch=True, on_step=True)
#        return val_loss
#    
#    def test_step(self, batch, batch_idx):
#        predicted_masks = self(pixel_values=batch["pixel_values"], input_boxes=batch["input_boxes"])
#        predicted_masks = torch.nn.functional.interpolate(predicted_masks,
#                size=(512, 512),
#                mode='bilinear',
#                align_corners=False)
#        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
#        test_loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
#
#        predicted_masks = predicted_masks.squeeze(1)
#
#        # Calculate IoU
#        intersection = torch.logical_and(predicted_masks > 0.8, ground_truth_masks > 0.5).float().sum((1, 2))
#        union = torch.logical_or(predicted_masks > 0.8, ground_truth_masks > 0.5).float().sum((1, 2))
#        iou = intersection / union
#
#
#        # Store results
#        self.log('test_loss', test_loss)
#        self.log('test_iou', iou.mean())
#        self.test_outputs.append({
#            'test_loss': test_loss,
#            'test_iou': iou,
#            'ground_truth_masks': ground_truth_masks.cpu().numpy(),
#            'predicted_masks': predicted_masks.cpu().numpy(),
#            'bboxes': batch["input_boxes"].cpu().numpy()
#        })
#        return {
#            'test_loss': test_loss,
#            'test_iou': iou,
#            'ground_truth_masks': ground_truth_masks.cpu().numpy(),
#            'predicted_masks': predicted_masks.cpu().numpy(),
#            'bboxes': batch["input_boxes"].cpu().numpy()
#        }
#
#    def on_test_epoch_end(self):
#        avg_test_loss = torch.stack([x['test_loss'] for x in self.test_outputs]).mean()
#        avg_test_iou = torch.cat([x['test_iou'] for x in self.test_outputs]).mean()
#        self.log('avg_test_loss', avg_test_loss)
#        self.log('avg_test_iou', avg_test_iou)
#
#        # Store ground truth and predicted masks
#        ground_truth_masks = [x['ground_truth_masks'] for x in self.test_outputs]
#        predicted_masks = [x['predicted_masks'] for x in self.test_outputs]
#        individual_ious = torch.cat([x['test_iou'] for x in self.test_outputs]).cpu().numpy()
#        bounding_boxes = [x['bboxes'] for x in self.test_outputs]
#        self.test_results = {
#            'ground_truth_masks': ground_truth_masks,
#            'predicted_masks': predicted_masks,
#            'individual_ious': individual_ious,
#            'bboxes': bounding_boxes
#        }
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
#        #optimizer = torch.optim.SGD(
#        #    filter(lambda p: p.requires_grad, self.parameters()),
#        #    lr=self.learning_rate,
#        #    momentum=0.9,      # Added momentum
#        #    weight_decay=1e-4  # Optional weight decay
#        #)
#        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
#        return {
#            'optimizer': optimizer,
#            'lr_scheduler': scheduler,
#            'monitor': 'val_loss'  # Metric to monitor for ReduceLROnPlateau
#        }
#    def compute_iou(self, pred_masks, gt_masks):
#        binary_pred_masks = (pred_masks > 0.5).float()
#        binary_gt_masks = (gt_masks > 0.5).float()
#        intersection = torch.logical_and(binary_pred_masks, binary_gt_masks).float().sum((1, 2))
#        union = torch.logical_or(binary_pred_masks, binary_gt_masks).float().sum((1, 2))
#        epsilon = 1e-8  # small constant to avoid division by zero
#        iou = intersection / (union + epsilon)
#        return iou.mean()
#    
#    def compute_ground_truth_masks(self, ground_truth, bounding_boxes, scale_factor=0.5):
#        ground_truth_masks = []
#        for i in range(ground_truth.shape[0]):
#            boxes = bounding_boxes[i]
#            for box in boxes:
#                gt_mask = torch.zeros((512, 512), dtype=torch.float32)
#                x, y, x2, y2 = [int(v) for v in (box * scale_factor).tolist()]  # Assuming boxes are in (x1, y1, x2, y2) format and need to be scaled
#                gt_mask[y:y2, x:x2] = ground_truth[i, y:y2, x:x2]
#                ground_truth_masks.append(gt_mask)
#        ground_truth_masks = torch.stack(ground_truth_masks)
#        return ground_truth_masks
#    
#
#class RandomRotation90:
#    def __call__(self, inputs):
#        img = inputs["pixel_values"]
#        mask = inputs["ground_truth_mask"]
#        boxes = inputs['input_boxes']
#        angle = np.random.choice([0, 90, 180, 270])
#        img = v2.functional.rotate(img, angle)
#        mask = v2.functional.rotate(mask, angle)
#        boxes = self.rotate_boxes(boxes, angle, img.shape[-2:])
#        inputs["pixel_values"] = img
#        inputs["ground_truth_mask"] = mask
#        inputs['input_boxes'] = boxes
#        return inputs
#
#    def rotate_boxes(self, boxes, angle, img_size):
#        w, h = img_size
#        if angle == 90:
#            return torch.tensor([[y1, w - x2, y2, w - x1] for x1, y1, x2, y2 in boxes.view(-1, 4)]).view(boxes.shape)
#        elif angle == 180:
#            return torch.tensor([[w - x2, h - y2, w - x1, h - y1] for x1, y1, x2, y2 in boxes.view(-1, 4)]).view(boxes.shape)
#        elif angle == 270:
#            return torch.tensor([[h - y2, x1, h - y1, x2] for x1, y1, x2, y2 in boxes.view(-1, 4)]).view(boxes.shape)
#        return boxes
#    
#class RandomFlip:
#    def __init__(self, p=0.66):
#        self.p = p
#
#    def __call__(self, inputs):
#        img = inputs["pixel_values"]
#        mask = inputs["ground_truth_mask"]
#        boxes = inputs['input_boxes'] if 'input_boxes' in inputs else None
#        slope = inputs['slope'] if 'slope' in inputs else None
#
#        if np.random.random() < self.p:
#            if np.random.choice([True, False]):
#                # Apply horizontal flip
#                img = v2.functional.hflip(img)
#                mask = v2.functional.hflip(mask)
#                if boxes is not None:
#                    boxes = self.flip_boxes_horizontally(boxes, img.shape[-1])
#                if slope is not None:
#                    slope = v2.functional.hflip(slope)
#            else:
#                # Apply vertical flip
#                img = v2.functional.vflip(img)
#                mask = v2.functional.vflip(mask)
#                if boxes is not None:
#                    boxes = self.flip_boxes_vertically(boxes, img.shape[-2])
#                if slope is not None:
#                    slope = v2.functional.vflip(slope)
#
#        inputs["pixel_values"] = img
#        inputs["ground_truth_mask"] = mask
#        if boxes is not None:
#            inputs['input_boxes'] = boxes
#        if slope is not None:
#            inputs['slope'] = slope
#        return inputs
#
#    def flip_boxes_horizontally(self, boxes, img_width):
#        return torch.tensor([[img_width - x2, y1, img_width - x1, y2] for x1, y1, x2, y2 in boxes.view(-1, 4)]).view(boxes.shape)
#
#    def flip_boxes_vertically(self, boxes, img_height):
#        return torch.tensor([[x1, img_height - y2, x2, img_height - y1] for x1, y1, x2, y2 in boxes.view(-1, 4)]).view(boxes.shape)
#    
#class RandomMasking:
#    def __init__(self, max_size=256, mask_value=0, max_masks=2, ):
#        self.max_size = max_size
#        self.mask_value = mask_value
#        self.max_masks = max_masks
#
#    def __call__(self, inputs):
#        img = inputs["pixel_values"]
#        boxes = inputs['input_boxes']
#
#        # Get image dimensions
#        _,_, h, w = img.shape
#
#        # Randomly choose the number of masks to apply
#        num_masks = np.random.randint(0, self.max_masks + 1)
#
#        for _ in range(num_masks):
#            # Randomly choose the size of the mask
#            mask_size = np.random.randint(1, self.max_size + 1)
#
#            # Randomly choose the top-left corner of the mask
#            top = np.random.randint(0, h - mask_size)
#            left = np.random.randint(0, w - mask_size)
#
#            # Check if the mask overlaps with any of the boxes
#            for index, box in enumerate(boxes):
#                overlaps = False
#                x1, y1, x2, y2 = box[0]
#                if not (left + mask_size < x1 or left > x2 or top + mask_size < y1 or top > y2):
#                    overlaps = True
#                    
#                if not overlaps:
#                    # Apply the mask to the image
#                    img[index, :, top:top + mask_size, left:left + mask_size] = self.mask_value
#
#        inputs["pixel_values"] = img
#        return inputs
#    
#class RandomNoise:
#    def __init__(self, noise_factor=0.01):
#        self.noise_factor = noise_factor
#
#    def __call__(self, inputs):
#        img = inputs["pixel_values"]
#
#        inputs["pixel_values"] = v2.GaussianNoise(clip=False, sigma = self.noise_factor)(img)
#
#        return inputs
#    
#class RandomAffine:
#    def __init__(self, max_translation=50):
#        """
#        Args:
#            max_translation (int): Maximum translation (in px) in both x and y directions.
#        """
#        self.max_translation = max_translation
#
#    def __call__(self, inputs):
#
#        img = inputs["pixel_values"]
#        mask = inputs["ground_truth_mask"]
#
#        # Choose random rotation angle
#        angle = np.random.uniform(0, 360)
#        # Choose random translation offsets (dx, dy)
#        tx = np.random.randint(-self.max_translation, self.max_translation)
#        ty = np.random.randint(-self.max_translation, self.max_translation)
#        # Use affine transform with angle rotation, scale=1, no shear.
#        img = v2.functional.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=0.0)
#        mask = v2.functional.affine(mask, angle=angle, translate=(tx, ty), scale=1.0, shear=0.0)
#
#        inputs["pixel_values"] = img
#        inputs["ground_truth_mask"] = mask
#        return inputs
#
#
#class SAMDataset(Dataset):
#  """
#  This class is used to create a dataset that serves input images and masks.
#  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
#  """
#  def __init__(self, dataset, augment = True, target_size=1024):
#    self.dataset = dataset
#    self.augment = augment
#    self.target_size = target_size
#
#  def __len__(self):
#    return len(self.dataset)
#
#  def __getitem__(self, idx):
#    item = self.dataset[idx]
#    image = np.array(item["image"])
#    ground_truth_mask = np.array(item["label"])
#    scale_factor = self.target_size / 512.0  # typically 2 when target is 1024
#
#    # get bounding box prompt
#    # tried -5 5, -5 15, -5 10
#    prompt = item["box"]
#    xchange = np.random.randint(-5, 5) if self.augment else 0
#    ychange = np.random.randint(-5, 5) if self.augment else 0
#    wchange = np.random.randint(-5, 5) if self.augment else 0
#    hchange = np.random.randint(-5, 5) if self.augment else 0
#    input_boxes = []
#    for box in prompt: 
#      x, y, w, h = box
#      input_boxes.append([[max(0, x - xchange), max(0, y - ychange), min(512, x + w + wchange), min(512, y + h + hchange)]])
#
#    ground_truth_mask = torch.from_numpy(ground_truth_mask).float()
#    image_tensor = torch.from_numpy(image).permute(0,3, 1, 2).float()  # Convert to tensor and change to (C, H, W)
#    boxes_tensor = torch.tensor(input_boxes, dtype=torch.float)
#
#    inputs = {
#            "pixel_values": image_tensor,
#            "ground_truth_mask": ground_truth_mask,
#            "input_boxes": boxes_tensor
#        }
#
#
#    if self.augment:
#        inputs = RandomRotation90()(inputs)
#        inputs = RandomFlip()(inputs)
#        inputs = RandomMasking()(inputs)
#        inputs = RandomNoise()(inputs)
#        
#    inputs["input_boxes"] = (inputs["input_boxes"]*scale_factor).squeeze(1)
#
#    return inputs
#  
#
#class SAMDataset3(Dataset):
#  """
#  This class is used to create a dataset that serves input images and masks.
#  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
#  """
#  def __init__(self, dataset, processor, do_normalize = False, do_rescale = False, augment = True, target_size=1024, type = "all", test = False):
#    self.dataset = dataset
#    self.processor = processor
#    self.normalize = do_normalize
#    self.rescale = do_rescale
#    self.augment = augment
#    self.target_size = target_size
#    self.type = type
#    self.test = test
#
#  def __len__(self):
#    return len(self.dataset)
#
#  def __getitem__(self, idx):
#    item = self.dataset[idx]
#    ground_truth_mask = np.array(item["label"])
#    VH0 = np.array(item["VH0"])
#    VH1 = np.array(item["VH1"])
#    VV0 = np.array(item["VV0"])
#    VV1 = np.array(item["VV1"])
#    dem = np.array(item["dem"])
#    slope = np.array(item["slope"])
#    scale_factor = self.target_size / 512.0  # typically 2 when target is 1024
#
#    if self.type == "VV":
#        image = np.stack([VV0, VV1, dem], axis=1)
#    elif self.type == "VH":
#        image = np.stack([VH0, VH1, dem], axis=1)
#    elif self.type == "combine":
#        a = _rescale(VH1 - VH0, 0, .25)
#
#        b = _rescale(VV1 - VV0, 0, .25)
#
#        w = _rescale(a - b, 0, 1)
#
#        r = w*VH0 + (1 - w)*VV0
#
#        g = w*VH1 + (1 - w)*VV1
#
#        image = np.stack([r, g, dem], axis=1)
#    else:
#        # Combine all channels to create image
#        image = np.stack([VH0, VH1, VV0, VV1, dem, slope], axis=1)
#
#    ground_truth_mask = torch.from_numpy(ground_truth_mask).float()
#    image_tensor = torch.from_numpy(image).float()  # Convert to tensor and change to (C, H, W)
#    
#    bounding_boxes = []
#
#    inputs = {
#            "pixel_values": image_tensor,
#            "ground_truth_mask": ground_truth_mask,
#        }
#    
#    if self.augment:
#        inputs = RandomAffine()(inputs)
#        inputs = RandomFlip()(inputs)
#        inputs = RandomNoise()(inputs)
#
#    if self.test:
#        prompt = item["box"]
#        for box in prompt:
#            x, y, w, h = box
#            bounding_boxes.append([[x, y, x + w, y + h]])
#    else:
#        for mask in inputs["ground_truth_mask"].numpy():
#            boxes = create_bounding_boxes(mask, augment=self.augment)
#
#            if len(boxes) == 0:
#                #print("No boxes found, using full image")
#                #boxes = np.array([[0, 0, 512, 512]])
#                # Instead of using the full image box, generate a random box:
#                # Define a minimum size for the random box, e.g., 50 pixels.
#                min_size = 50
#                # Random width and height between min_size and 512.
#                w_random = np.random.randint(min_size, 512)
#                h_random = np.random.randint(min_size, 512)
#                # Ensure the box is within boundaries.
#                x_random = np.random.randint(0, 512 - w_random)
#                y_random = np.random.randint(0, 512 - h_random)
#                boxes = np.array([[x_random, y_random, w_random, h_random]])
#            else:
#                boxes = np.array(boxes)
#            #transform boxes to (x1, y1, x2, y2) from (x, y, w, h)
#            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
#            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
#            bounding_boxes.append(boxes)
#
#    for i in range(len(bounding_boxes)):
#        bounding_boxes[i] = torch.tensor(bounding_boxes[i], dtype=torch.float)
#
#    inputs["input_boxes"] = [box * scale_factor for box in bounding_boxes] 
#    inputs["pixel_values"] = torch.nn.functional.interpolate(inputs["pixel_values"], size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
#
#    return inputs