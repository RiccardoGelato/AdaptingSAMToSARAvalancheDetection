import torch
import pytorch_lightning as pl
import torch.nn as nn
import monai
from sam.build_sam import sam_model_registry
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sam.build_sam import sam_model_registry
from typing import List
from datasets import Dataset
from torchvision.transforms import v2
from model.convNet import AutoEncoderSmall, PretrainedAutoEncoder, AutoEncoderMultiDim
from model.baseModel import BaseSamModel
from model.inputTypes import InputTypes


class AutoSAM(nn.Module):

    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        normalize = True,
        pixel_mean: List[float] = [0.5183216317041373, 0.5139202875277856, 0.1656409264909524], #normalMean
        pixel_std: List[float] = [0.2547065818875149, 0.2571921631489484, 0.06727353387015675],#normalStd
        pixel_mean_final: List[float] = [0.4224749803543091, 0.3967047929763794, 0.5431008338928223, 0.5183878540992737, 0.16572988033294678, 0.24865785241127014], #normalMean
        pixel_std_final: List[float] = [0.23271377384662628, 0.2313859462738037, 0.2603246569633484, 0.2645578682422638, 0.06731900572776794, 0.15417179465293884], #normalStd
        pixel_mean_VV: List[float] = [0.5431008338928223, 0.5183878540992737, 0.16572988033294678], #VVMean
        pixel_std_VV: List[float] = [0.2603246569633484, 0.2645578682422638, 0.06731900572776794], #VVStd
        pixel_mean_VH: List[float] = [0.4224749803543091, 0.3967047929763794, 0.24865785241127014], #VHMean
        pixel_std_VH: List[float] = [0.23271377384662628, 0.2313859462738037, 0.15417179465293884], #VHStd
        input_type: InputTypes = InputTypes.Normal
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
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
        self.normalize = normalize
        self.autoencoder = AutoEncoderSmall() #PretrainedAutoEncoder() #AutoEncoderSmall() #AutoEncoderMultiDim(3)
    
    def forward(self, pixel_values, input_boxes, points=None, masks=None, slopes=None, meteo_data=None):
    
        outputs = []
        batch_size = pixel_values.shape[0]
        
        # Optionally, apply normalization once for each image.
        if self.normalize:
            pixel_values = self.preprocess(pixel_values)

        image_embeddings = self.image_encoder(pixel_values)  # (B, 256, 64, 64)

        image_embeddings = image_embeddings + self.autoencoder(pixel_values)  # (B, 256, 64, 64)

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
    def _build_model(self, image_encoder, mask_decoder, prompt_encoder, normalize=True, adapt_patch_embed=False, input_type: str = "normal") :
        return AutoSAM(
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
#    def __init__(self, model_name, learning_rate=1e-3, startOver=True, normalize=True, encoder=None, adapt=False):
#        super(LitSamModel, self).__init__()
#
#        if model_name == "vit_l":
#            model = sam_model_registry["vit_l"]("sam_vit_l_0b3195.pth", adapt = adapt)
#        else:
#            model = sam_model_registry["vit_b"]("/home/gelato/Avalanche-Segmentation-with-Sam/code/model/sam_vit_b_01ec64.pth", adapt = adapt)
#        
#
#        if encoder is None:
#            self.model = AutoSAM(model.image_encoder, model.mask_decoder, model.prompt_encoder, normalize=normalize)  
#        else:
#            self.model = AutoSAM(encoder, model.mask_decoder, model.prompt_encoder, normalize=normalize) 
#
#        # Freeze specific weights
#        for name, param in self.model.named_parameters():
#            if name.startswith("image_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
#                #if "Adapter" not in name:
#                    param.requires_grad = False 
#
#        self.learning_rate = learning_rate
#        #Try DiceFocalLoss, FocalLoss, DiceCELoss
#        self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
#        #self.seg_loss = monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')
#        #self.seg_loss = monai.losses.FocalLoss()
#        self.test_outputs = []  # Initialize the test_outputs attribute
#
#
#    def forward(self, pixel_values, input_boxes):
#        return self.model(pixel_values=pixel_values, input_boxes=input_boxes)
#    
#    @classmethod
#    def load_from_checkpoint_strictless(cls, checkpoint_path, **kwargs):
#        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
#        # Initialize model (this calls __init__)
#        model = cls(**kwargs)
#        # Load checkpoint state dict without requiring all keys
#        model.load_state_dict(checkpoint["state_dict"], strict=False)
#        return model
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
#        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
#        return {
#            'optimizer': optimizer,
#            'lr_scheduler': scheduler,
#            'monitor': 'val_loss'  # Metric to monitor for ReduceLROnPlateau
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