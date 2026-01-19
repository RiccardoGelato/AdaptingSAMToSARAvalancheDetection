import torch
import pytorch_lightning as pl
import torch.nn as nn
import monai
from sam.build_sam import sam_model_registry
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sam.build_sam import sam_model_registry
from typing import List, Optional
from datasets import Dataset
from torchvision.transforms import v2
import lora.loraModules as lora
from model.baseModel import BaseSamModel, AdaptedSAM
from model.imageAugmentations import RandomRotation90, RandomFlip, RandomMasking, RandomNoise, RandomAffine, process_meteo
from dataprocessing.rcsHandlingFunctions import _rescale
from dataprocessing.creationOfDataframe import create_bounding_boxes
from model.inputTypes import InputTypes

class LitSamModel(BaseSamModel):
    def _build_model(self, image_encoder, mask_decoder, prompt_encoder, normalize=True, adapt_patch_embed=False, input_type: InputTypes = InputTypes.Normal, rank: Optional[int]=None, alpha: Optional[int]=None) :

        model = AdaptedSAM(
            image_encoder=image_encoder,
            mask_decoder=mask_decoder,
            prompt_encoder=prompt_encoder,
            normalize=normalize,
            input_type=input_type
        )
        # Example usage for AdaptedSAM (or in the image_encoder):
        lora.replace_linear_with_lora(model.image_encoder, target_layer_names=["qkv", "proj"], r=rank, alpha=alpha)
        return model

    def should_freeze(self, name, param):
        # Freeze image and prompt encoder except Adapters and LoRA layers

        if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
                if "lora" not in name :#and "proj" not in name and "qkv" not in name:
                    return True
                #if "patch_embed" in name:
                #    param.requires_grad = False
        return False
    
#
#class LitSamModel(pl.LightningModule):
#    def __init__(self, model_name, learning_rate=1e-3, normalize=True, encoder=None, rank = 128, alpha = 32.0, adapt=True):
#        super(LitSamModel, self).__init__()
#
#        if model_name == "vit_l":
#            model = sam_model_registry["vit_l"]("sam_vit_l_0b3195.pth")
#        else:
#            model = sam_model_registry["vit_b"]("/home/gelato/Avalanche-Segmentation-with-Sam/code/model/sam_vit_b_01ec64.pth")
#        
#
#        if encoder is None:
#            self.model = AdaptedSAM(model.image_encoder, model.mask_decoder, model.prompt_encoder, normalize=normalize)  
#        else:
#            self.model = AdaptedSAM(encoder, model.mask_decoder, model.prompt_encoder, normalize=normalize) 
#
#        # Example usage for AdaptedSAM (or in the image_encoder):
#        lora.replace_linear_with_lora(self.model.image_encoder, target_layer_names=["qkv", "proj"], r=rank, alpha=alpha)
#
#        # Freeze specific weights
#        for name, param in self.model.named_parameters():
#            if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
#                if "lora" not in name :#and "proj" not in name and "qkv" not in name:
#                    param.requires_grad = False
#                #if "patch_embed" in name:
#                #    param.requires_grad = False
#
#        self.learning_rate = learning_rate
#        #Try DiceFocalLoss, FocalLoss, DiceCELoss
#        self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
#        #self.seg_loss = monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')
#        self.test_outputs = []  # Initialize the test_outputs attribute
#        self.rank = rank
#        self.alpha = alpha
#
#    def forward(self, pixel_values, input_boxes):
#        return self.model(pixel_values=pixel_values, input_boxes=input_boxes)
#
#    def training_step(self, batch, batch_idx):
#        predicted_masks = self(pixel_values=batch["pixel_values"], input_boxes=batch["input_boxes"])
#        predicted_masks = torch.nn.functional.interpolate(predicted_masks,
#                size=(512, 512),
#                mode='bilinear',
#                align_corners=False)
#        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
#        loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
#        iou = self.compute_iou(predicted_masks.squeeze(1), ground_truth_masks)
#        self.log('train_iou', iou, on_epoch=True, on_step=True)
#        self.log('train_loss', loss, on_epoch=True, on_step=True)
#        return loss
#    
#    def validation_step(self, batch, batch_idx):
#        predicted_masks = self(pixel_values=batch["pixel_values"], input_boxes=batch["input_boxes"])
#        predicted_masks = torch.nn.functional.interpolate(predicted_masks,
#                size=(512, 512),
#                mode='bilinear',
#                align_corners=False)
#        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
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
#        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)
#        return {
#            'optimizer': optimizer,
#            'lr_scheduler': scheduler,
#            'monitor': 'val_iou'  # Metric to monitor for ReduceLROnPlateau
#        }
#    
#    def compute_iou(self, pred_masks, gt_masks):
#        binary_pred_masks = (pred_masks > 0.5).float()
#        binary_gt_masks = (gt_masks > 0.5).float()
#        intersection = torch.logical_and(binary_pred_masks, binary_gt_masks).float().sum((1, 2))
#        union = torch.logical_or(binary_pred_masks, binary_gt_masks).float().sum((1, 2))
#        epsilon = 1e-8  # small constant to avoid division by zero
#        iou = intersection / (union + epsilon)
#        return iou.mean()
#    
#    def on_save_checkpoint(self, checkpoint):
#        # Add custom keys to the checkpoint
#        checkpoint['rank'] = self.rank
#        checkpoint['alpha'] = self.alpha
#        return checkpoint