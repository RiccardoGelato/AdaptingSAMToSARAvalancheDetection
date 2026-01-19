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
from sam.modeling.common import LayerNorm2d
from model.imageAugmentations import RandomRotation90, RandomFlip, RandomMasking, RandomNoise, RandomAffine, process_meteo
from dataprocessing.creationOfDataframe import create_bounding_boxes
from dataprocessing.rcsHandlingFunctions import _rescale
from model.inputTypes import InputTypes
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


#class SlopePromptEncoder(nn.Module):
#    def __init__(self, embed_dim, slope_input_channels=1):
#        super().__init__()
#        self.conv = nn.Sequential(
#            nn.Conv2d(slope_input_channels, embed_dim // 4, kernel_size=3, stride=2, padding=1),
#            LayerNorm2d(embed_dim // 4),
#            nn.ReLU(),
#            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
#            LayerNorm2d(embed_dim // 2),
#            nn.ReLU(),
#            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),            
#        )
#
#    def forward(self, slope):
#        # slope: (B, slope_input_channels, H, W)
#        x = self.conv(slope)       # (B, embed_dim, H/8, W/8)
#        return x                   # (B, embed_dim, H/8, W/8)
#
#class MeteoEncoderRNN(nn.Module):
#    def __init__(self, input_feature_dim: int, embed_dim: int, hidden_size: int = 64, num_layers: int = 1):
#        super().__init__()
#        self.lstm = nn.LSTM(input_feature_dim, hidden_size, num_layers=num_layers, batch_first=True)
#        self.fc = nn.Linear(hidden_size, embed_dim)
#        
#    def forward(self, meteo_data):
#        # meteo_data: (B, T, input_feature_dim)
#        # Process with LSTM: outputs (B, T, hidden_size)
#        outputs, (h_n, _) = self.lstm(meteo_data)
#        # Use the last hidden state of the last LSTM layer (B, hidden_size)
#        last_hidden = h_n[-1]
#        # Convert to embedding dim (B, embed_dim)
#        embedding = self.fc(last_hidden)
#        return embedding
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            LayerNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)
    
class SlopeMeteoEncoder(nn.Module):
    def __init__(self,input_meteo_dim: int,meteo_embed_dim: int, embed_dim: int, hidden_size: int = 64, num_layers: int = 1, input_channels=1):
        super().__init__()
        # Normalize the meteo input before LSTM: normalize over the last dimension (features)
        self.input_norm = nn.LayerNorm(input_meteo_dim)
        self.lstm = nn.LSTM(input_meteo_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, meteo_embed_dim)
        self.convBlock1 = ConvBlock(input_channels, embed_dim // 4, kernel_size=3, stride=2, padding=1)
        self.convBlock2 = ConvBlock(embed_dim // 4 + meteo_embed_dim, embed_dim // 2, kernel_size=3, stride=2, padding=1)
        self.finalConv = nn.Conv2d(embed_dim // 2 + meteo_embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        # Add a normalization layer to ensure low-magnitude outputs
        self.meteo_norm = nn.LayerNorm(meteo_embed_dim)

        self.register_buffer("meteo_mean", torch.tensor([270.591, 0.273, 7.104, 0.860, 100348.156], dtype=torch.float))
        self.register_buffer("meteo_std", torch.tensor([3.743, 0.684, 4.419, 0.1349, 1650.618], dtype=torch.float))

        self._initialize_weights()
        

    def forward(self, slope, meteo_data):
        # meteo_data: (B, T, input_feature_dim)
        # Manually normalize over the last dimension:
        meteo_data = (meteo_data - self.meteo_mean) / self.meteo_std
        # Normalize input before LSTM
        meteo_data = self.input_norm(meteo_data)
        # Process with LSTM: outputs (B, T, hidden_size)
        outputs, (h_n, _) = self.lstm(meteo_data)
        # Use the last hidden state of the last LSTM layer (B, hidden_size)
        last_hidden = h_n[-1]

        # Convert to embedding dim (B, meteo_embed_dim)
        meteo_embedding = self.fc(last_hidden)  # (B, meteo_embed_dim)
        meteo_embedding = self.meteo_norm(meteo_embedding)

        prompt = self.convBlock1(slope)       # (B, embed_dim // 4, H/2, W/2)
        
        meteo_repeated = self.expand_meteo(meteo_embedding, prompt.shape)  # (B, meteo_embed_dim, H/2, W/2)

        prompt = torch.cat((prompt, meteo_repeated), dim=1)  # Concatenate along channel dimension
        prompt = self.convBlock2(prompt)  # (B, embed_dim // 2, H/4, W/4)

        meteo_repeated = self.expand_meteo(meteo_embedding, prompt.shape)  # (B, meteo_embed_dim, H/4, W/4)
        prompt = torch.cat((prompt, meteo_repeated), dim=1)  # Concatenate along channel dimension
        prompt = self.finalConv(prompt)  # (B, embed_dim, H/8, W/8)
        return prompt, meteo_embedding  # (B, embed_dim, H/8, W/8), (B, meteo_embed_dim)
    
    def _initialize_weights(self):
        # Initialize LSTM weights for low outputs:
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                # Use Xavier with a small gain
                nn.init.xavier_uniform_(param, gain=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
        ## Initialize fc layer with small weights
        #nn.init.xavier_uniform_(self.fc.weight, gain=0.01)
        #if self.fc.bias is not None:
        #    nn.init.constant_(self.fc.bias, 0)
        
        # Initialize the convolution blocks with a small gain
        #for conv in [self.convBlock1.conv[0], self.convBlock2.conv[0], self.finalConv]:
        #    nn.init.xavier_uniform_(conv.weight, gain=0.01)
        #    if conv.bias is not None:
        #        nn.init.constant_(conv.bias, 0)

    def expand_meteo(self, meteo_embedding, prompt_shape):
        # Repeat meteo embedding to match spatial dimensions of prompt
        B, _, H, W = prompt_shape
        meteo_repeated = meteo_embedding.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
        # Now meteo_repeated has shape: (B, meteo_embed_dim, H/2, W/2)
        return meteo_repeated
        

class AdaptedSAM(nn.Module):
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
        #pixel_mean_VH: List[float] = [0.4224749803543091, 0.3967047929763794, 0.16572988033294678], #VHMean
        #pixel_std_VH: List[float] = [0.23271377384662628, 0.2313859462738037, 0.06731900572776794], #VHStd
        pixel_mean_dif: List[float] = [-0.024642864242196083, -0.025667885318398476, 0.1658152937889099], #DIFMean
        pixel_std_dif: List[float] = [0.12830990552902222, 0.1322723925113678, 0.06732339411973953], #DIFStd
        pixel_mean_original: List[float] = [123.675, 116.28, 103.53],
        pixel_std_original: List[float] = [58.395, 57.12, 57.375],
        input_type: InputTypes = InputTypes.Normal #Options: Normal, All, VV, VH
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
        elif input_type == InputTypes.Original:
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean_original).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std_original).view(-1, 1, 1), False)
        elif input_type == InputTypes.Difference:
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean_dif).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std_dif).view(-1, 1, 1), False)
        else:
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.normalize = normalize
        #self.slope_encoder = SlopePromptEncoder(embed_dim=256, slope_input_channels=1)
        #self.meteo_encoder = SlopeMeteoEncoder(input_meteo_dim=5, meteo_embed_dim=256, embed_dim=256, hidden_size=64, num_layers=1, input_channels=1)
    
    def forward(self, pixel_values, input_boxes, points=None, masks=None, slopes=None, meteo_data=None):
        
        #if self.normalize:
        #    pixel_values = self.preprocess(pixel_values)
        #
        #image_embeddings = self.image_encoder(pixel_values)  # (B, 256, 64, 64)
        #
        #with torch.no_grad(): #Avoid gradient computation for prompt_encoder
        #    sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #                    points=None,
        #                    boxes=input_boxes,
        #                    masks=None,
        #                )
        ##slope_embeddings = self.slope_encoder(slopes) if slopes is not None else torch.zeros(dense_embeddings.shape, device=pixel_values.device)
        ##if slopes is not None and meteo_data is not None:
        ##    meteo_prompt, meteo_embedding = self.meteo_encoder(slopes, meteo_data)  
        ##else:
        ##    meteo_prompt = torch.zeros(dense_embeddings.shape, device=pixel_values.device)
        ##meteo_prompt = torch.zeros(dense_embeddings.shape, device=pixel_values.device)
        #low_res_masks, _ = self.mask_decoder(
        #    image_embeddings=image_embeddings,  # (B, 256, 64, 64)
        #    image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        #    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        #    dense_prompt_embeddings=dense_embeddings, #+ meteo_prompt,  # (B, 256, 64, 64)
        #    multimask_output=False,
        #)
    
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
    
#    def forward(self, pixel_values, input_boxes, points=None, masks=None, slopes=None, meteo_data=None):
#        batch_size = pixel_values.shape[0]
#
#        # Normalize pixel values if required
#        if self.normalize:
#            pixel_values = self.preprocess(pixel_values)
#
#        # Get image embeddings for the entire batch
#        image_embeddings = self.image_encoder(pixel_values)  # Shape: (B, 256, 64, 64)
#
#        # Flatten bounding boxes across the batch
#        all_boxes = []
#        image_indices = []
#        for i in range(batch_size):
#            if len(input_boxes[i].shape) == 1:
#                # If only one box per image, add a dimension
#                input_boxes[i] = input_boxes[i].unsqueeze(0)  # Shape: (1, 4)
#            all_boxes.append(input_boxes[i])
#            image_indices.extend([i] * input_boxes[i].shape[0])  # Track which image each box belongs to
#
#        all_boxes = torch.cat(all_boxes, dim=0)  # Shape: (total_boxes, 4)
#        image_indices = torch.tensor(image_indices, device=all_boxes.device)  # Shape: (total_boxes,)
#
#        # Process prompts for all bounding boxes
#        with torch.no_grad():
#            sparse_embeddings, dense_embeddings = self.prompt_encoder(
#                points=None,
#                boxes=all_boxes,
#                masks=None,
#            )
#
#        # Expand image embeddings to match the number of bounding boxes
#        expanded_image_embeddings = image_embeddings[image_indices]  # Shape: (total_boxes, 256, 64, 64)
#
#        # Decode masks for all bounding boxes
#        low_res_masks, _ = self.mask_decoder(
#            image_embeddings=expanded_image_embeddings,         # Shape: (total_boxes, 256, 64, 64)
#            image_pe=self.prompt_encoder.get_dense_pe(),        # Shape: (1, 256, 64, 64) or broadcastable
#            sparse_prompt_embeddings=sparse_embeddings,         # Shape: (total_boxes, 2, 256)
#            dense_prompt_embeddings=dense_embeddings,           # Shape: (total_boxes, 256, 64, 64)
#            multimask_output=False,
#        )
#
#        # Reshape the output masks back to the original batch structure
#        outputs = []
#        start_idx = 0
#        for i in range(batch_size):
#            num_boxes = (image_indices == i).sum().item()
#            outputs.append(low_res_masks[start_idx:start_idx + num_boxes])  # Collect masks for this image
#            start_idx += num_boxes
#
#        outputs = torch.cat(outputs, dim=0)
#
#        return outputs  
    
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


class LitSamModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate=1e-3, startOver=True, normalize=True, encoder=None, decoder=None, increase_resolution=False, adapt=True, adapt_patch_embed=False, mlp_ratio_adapter=0.25, input_type: InputTypes = InputTypes.Normal, HQ=False, IR=False):
        super(LitSamModel, self).__init__()

        if model_name == "vit_l":
            model = sam_model_registry["vit_b"](os.path.join(SAM_CHECKPOINT, "sam_vit_l_0b3195.pth"), adapt = adapt, increase_resolution=increase_resolution, adapt_patch_embed=adapt_patch_embed, mlp_ratio_adapter=mlp_ratio_adapter, HQ=HQ, IR=IR)
        else:
            model = sam_model_registry["vit_b"](os.path.join(SAM_CHECKPOINT, "sam_vit_b_01ec64.pth"), adapt = adapt, increase_resolution=increase_resolution, adapt_patch_embed=adapt_patch_embed, mlp_ratio_adapter=mlp_ratio_adapter, HQ=HQ, IR=IR)

        if encoder is None:
            if decoder is None:
                self.model = AdaptedSAM(model.image_encoder, model.mask_decoder, model.prompt_encoder, normalize=normalize, input_type=input_type, )  
            else:
                self.model = AdaptedSAM(model.image_encoder, decoder, model.prompt_encoder, normalize=normalize, input_type=input_type)
        else:
            if decoder is None:
                self.model = AdaptedSAM(encoder, model.mask_decoder, model.prompt_encoder, normalize=normalize, input_type=input_type)
            else:
                self.model = AdaptedSAM(encoder, decoder, model.prompt_encoder, normalize=normalize, input_type=input_type)
             

        # Freeze specific weights
        for name, param in self.model.named_parameters():
            if name.startswith("image_encoder") or name.startswith("prompt_encoder"): #or name.startswith("mask_decoder"):
                if "Adapter" not in name:
                    param.requires_grad = False
        # Define class weights: [background, avalanche]
        class_weights = torch.tensor([2.0], device=self.device)
        self.learning_rate = learning_rate
        #Try DiceFocalLoss, DiceCELoss, DiceLoss
        self.seg_loss = monai.losses.DiceLoss(sigmoid = True, squared_pred=True)
        #self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean', weight = class_weights)
        #self.seg_loss = monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean', weight = class_weights)
        self.test_outputs = []  # Initialize the test_outputs attribute

    def forward(self, pixel_values, input_boxes, slopes=None, meteo_data = None):
        return self.model(pixel_values=pixel_values, input_boxes=input_boxes, slopes =slopes, meteo_data=meteo_data)
    
    @classmethod
    def load_from_checkpoint_strictless(cls, checkpoint_path, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # Initialize model (this calls __init__)
        model = cls(**kwargs)
        # Load checkpoint state dict without requiring all keys
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        return model

    def training_step(self, batch, batch_idx):
        predicted_masks = self(pixel_values=batch["pixel_values"], input_boxes=batch["input_boxes"], slopes=batch["slope"] if "slope" in batch else None, meteo_data=batch["meteo"] if "meteo" in batch else None)
        predicted_masks = torch.nn.functional.interpolate(predicted_masks,
                size=(512, 512),
                mode='bilinear',
                align_corners=False)
        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
        ground_truth_masks = self.compute_ground_truth_masks(ground_truth_masks, batch["input_boxes"]).to(self.device)

        loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
        iou = self.compute_iou(predicted_masks.squeeze(1), ground_truth_masks)
        self.log('train_iou', iou, on_epoch=True, on_step=True)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        predicted_masks = self(pixel_values=batch["pixel_values"], input_boxes=batch["input_boxes"], slopes=batch["slope"] if "slope" in batch else None)
        predicted_masks = torch.nn.functional.interpolate(predicted_masks,
                size=(512, 512),
                mode='bilinear',
                align_corners=False)
        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)

        ground_truth_masks = self.compute_ground_truth_masks(ground_truth_masks, batch["input_boxes"]).to(self.device)

        val_loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
        iou = self.compute_iou(predicted_masks.squeeze(1), ground_truth_masks)
        self.log('val_iou', iou, on_epoch=True, on_step=True)
        self.log('val_loss', val_loss, on_epoch=True, on_step=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        predicted_masks = self(pixel_values=batch["pixel_values"], input_boxes=batch["input_boxes"], slopes=batch["slope"] if "slope" in batch else None, meteo_data=batch["meteo"] if "meteo" in batch else None)
        predicted_masks = torch.nn.functional.interpolate(predicted_masks,
                size=(512, 512),
                mode='bilinear',
                align_corners=False)
        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
        test_loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        predicted_masks = predicted_masks.squeeze(1)

        # Calculate IoU
        intersection = torch.logical_and(predicted_masks > 0.8, ground_truth_masks > 0.5).float().sum((1, 2))
        union = torch.logical_or(predicted_masks > 0.8, ground_truth_masks > 0.5).float().sum((1, 2))
        iou = intersection / union


        # Store results
        self.log('test_loss', test_loss)
        self.log('test_iou', iou.mean())
        self.test_outputs.append({
            'test_loss': test_loss,
            'test_iou': iou,
            'ground_truth_masks': ground_truth_masks.cpu().numpy(),
            'predicted_masks': predicted_masks.cpu().numpy(),
            'bboxes': batch["input_boxes"].cpu().numpy()
        })
        return {
            'test_loss': test_loss,
            'test_iou': iou,
            'ground_truth_masks': ground_truth_masks.cpu().numpy(),
            'predicted_masks': predicted_masks.cpu().numpy(),
            'bboxes': batch["input_boxes"].cpu().numpy()
        }

    def on_test_epoch_end(self):
        avg_test_loss = torch.stack([x['test_loss'] for x in self.test_outputs]).mean()
        avg_test_iou = torch.cat([x['test_iou'] for x in self.test_outputs]).mean()
        self.log('avg_test_loss', avg_test_loss)
        self.log('avg_test_iou', avg_test_iou)

        # Store ground truth and predicted masks
        ground_truth_masks = [x['ground_truth_masks'] for x in self.test_outputs]
        predicted_masks = [x['predicted_masks'] for x in self.test_outputs]
        individual_ious = torch.cat([x['test_iou'] for x in self.test_outputs]).cpu().numpy()
        bounding_boxes = [x['bboxes'] for x in self.test_outputs]
        losses = [x['test_loss'].cpu().numpy() for x in self.test_outputs]
        self.test_results = {
            'ground_truth_masks': ground_truth_masks,
            'predicted_masks': predicted_masks,
            'individual_ious': individual_ious,
            'bboxes': bounding_boxes,
            'losses': losses
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        #optimizer = torch.optim.SGD(
        #    filter(lambda p: p.requires_grad, self.parameters()),
        #    lr=self.learning_rate,
        #    momentum=0.9,      # Added momentum
        #    weight_decay=1e-4  # Optional weight decay
        #)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,90], gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_iou'  # Metric to monitor for ReduceLROnPlateau
        }
    def compute_iou(self, pred_masks, gt_masks):
        binary_pred_masks = (pred_masks > 0.5).float()
        binary_gt_masks = (gt_masks > 0.5).float()
        intersection = torch.logical_and(binary_pred_masks, binary_gt_masks).float().sum((1, 2))
        union = torch.logical_or(binary_pred_masks, binary_gt_masks).float().sum((1, 2))
        epsilon = 1e-8  # small constant to avoid division by zero
        iou = intersection / (union + epsilon)
        return iou.mean()
    
    def compute_ground_truth_masks(self, ground_truth, bounding_boxes, scale_factor=0.5):
        ground_truth_masks = []
        for i in range(ground_truth.shape[0]):
            boxes = bounding_boxes[i]
            for box in boxes:
                gt_mask = torch.zeros((512, 512), dtype=torch.float32)
                x, y, x2, y2 = [int(v) for v in (box * scale_factor).tolist()]  # Assuming boxes are in (x1, y1, x2, y2) format and need to be scaled
                gt_mask[(y):(y2), (x):(x2)] = ground_truth[i, (y):(y2), (x):(x2)]
                ground_truth_masks.append(gt_mask)
        ground_truth_masks = torch.stack(ground_truth_masks)
        return ground_truth_masks



  

