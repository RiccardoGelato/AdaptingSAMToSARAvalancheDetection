from pytest import param
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
from model.baseModel import BaseSamModel
from model.imageAugmentations import RandomRotation90, RandomFlip, RandomMasking, RandomNoise, RandomAffine, process_meteo
from dataprocessing.rcsHandlingFunctions import _rescale
from dataprocessing.creationOfDataframe import create_bounding_boxes
from model.sfg import SelectiveFusionGate
from model.convNet import AutoEncoderSmall, AutoEncoderMultiDim


class SumSAM(nn.Module):
    def __init__(
        self,
        image_encoders,
        mask_decoder,
        prompt_encoder,
        coefficients,
        pixel_meanVV: List[float] = [0.535663902759552, 0.5097837448120117, 0.16753901541233063], #VVMean
        pixel_stdVV: List[float] = [0.266330361366272, 0.26998206973075867, 0.06870843470096588] ,#VVStd
        pixel_meanVH: List[float] = [0.4224749803543091, 0.3967047929763794, 0.24865785241127014], #VHMean
        pixel_stdVH: List[float] = [0.23271377384662628, 0.2313859462738037, 0.15417179465293884], #VHStd
        normalize = True,
    ):
        super().__init__()
        self.image_encoders = nn.ModuleList(image_encoders)
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.register_buffer("pixel_meanVV", torch.Tensor(pixel_meanVV).view(-1, 1, 1), False)
        self.register_buffer("pixel_stdVV", torch.Tensor(pixel_stdVV).view(-1, 1, 1), False)
        self.register_buffer("pixel_meanVH", torch.Tensor(pixel_meanVH).view(-1, 1, 1), False)
        self.register_buffer("pixel_stdVH", torch.Tensor(pixel_stdVH).view(-1, 1, 1), False)
        self.normalize = normalize
        self.coefficients = coefficients
        #self.sfg = SelectiveFusionGate(in_channels_per_feat=256, filter_num=1, feat_num=len(image_encoders), intermediate_channels=[32], filter_type='conv2d')
        #self.skipencoder = AutoEncoderSmall(num_classes = 768, input_channels = 6)
    
    def forward(self, pixel_values, input_boxes):

        outputs = []
        batch_size = pixel_values.shape[0]

        if self.normalize:
            pixel_values[0] = self.preprocess(pixel_values[0],index=0)
            pixel_values[1] = self.preprocess(pixel_values[1],index=1)
        
        ###pixel_values_concatenated = torch.cat((pixel_values[0], pixel_values[1]), dim=1) # (B, 6, H, W)
        
        #interm_embeddings = self.skipencoder(pixel_values_concatenated) # (B, 768, 64, 64), trying to skip to the decoder with an indipendent net
        # Convert from (B, H, W, C) to (B, C, H, W)
        #interm_embeddings = interm_embeddings.permute(0, 2, 3, 1)

        image_embeddings = []   
        for i, encoder in enumerate(self.image_encoders):
            # compute gradients only if encoder has trainable params
            requires = any(p.requires_grad for p in encoder.parameters())
            with torch.set_grad_enabled(requires):
                emb = encoder(pixel_values[i])
            # if frozen, detach to avoid keeping any autograd graph / extra memory
            if not requires:
                emb = emb.detach()
            image_embeddings.append(emb)  # (B, 256, 64, 64)

        #image_embeddings, fuse_feat_masks = self.sfg(image_embeddings) # (B, 256, 64, 64)
        image_embeddings = image_embeddings[0] * self.coefficients[0] + image_embeddings[1] * self.coefficients[1]

        for i in range(batch_size):
            if len(input_boxes[0].shape) == 1:
                # If only one box per image, add a dimension
                input_boxes = input_boxes.unsqueeze(1) #from (B, 4) to (B, 1, 4)
            bounding_boxes = input_boxes[i]
            number_of_boxes = bounding_boxes.shape[0]
            embedding_of_image = image_embeddings[i].unsqueeze(0)  # (1, 256, 64, 64)
            #interm_embedding_of_image = [interm_embeddings[i].unsqueeze(0)]  # (1, 768, 64, 64) list added for code consistency
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
                    #interm_embeddings=interm_embedding_of_image,
                    #hq_token_only = False,
                )
                outputs.append(low_res_masks)
        
        low_res_masks = torch.cat(outputs, dim=0)
        
        return low_res_masks
    
    def preprocess(self, x: torch.Tensor, index = 0) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if index == 0:
            pixel_mean = self.pixel_meanVV
            pixel_std = self.pixel_stdVV
        else:
            pixel_mean = self.pixel_meanVH
            pixel_std = self.pixel_stdVH
        # Normalize colors
        x = (x - pixel_mean) / pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoders[index].img_size - h
        padw = self.image_encoders[index].img_size - w
        x = nn.functional.pad(x, (0, padw, 0, padh))
        return x


class SumSamModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate=1e-3, startOver=True, normalize=True, HQ = False, alternate_every_n_epochs: int = 1):
        super(SumSamModel, self).__init__()

        if model_name == "vit_l":
            model = sam_model_registry["vit_l"]("sam_vit_l_0b3195.pth", adapt = True)
        else:
            model = sam_model_registry["vit_b"]("../model/sam_vit_b_01ec64.pth", adapt = True, HQ=HQ)
            model2 = sam_model_registry["vit_b"]("../model/sam_vit_b_01ec64.pth", adapt = True, HQ=HQ)

        image_encoders = [model.image_encoder, model2.image_encoder]

        if not HQ:
            self.model = SumSAM(image_encoders, model.mask_decoder, model.prompt_encoder, normalize=normalize, coefficients=[0.5,0.5]) 
    
            # Freeze specific weights
            for name, param in self.model.named_parameters():
                if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
                    if "Adapter" not in name:
                        param.requires_grad = False
        else:
            self.model = SumSAM(image_encoders, model.mask_decoder, model.prompt_encoder, normalize=normalize, coefficients=[0.5,0.5]) 
            # Freeze specific weights
            for name, param in self.model.named_parameters():
                #if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
                if any(hq_key in name for hq_key in ['hf_token', 'hf_mlp', 'compress_vit_feat', 'embedding_encoder', 'embedding_maskfeature', 'skipencoder', 'sfg']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.learning_rate = learning_rate
        #Try DiceFocalLoss, FocalLoss, DiceCELoss
        self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        #self.seg_loss = monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.test_outputs = []  # Initialize the test_outputs attribute

        # Alternating encoder training settings
        self.alternate_every_n_epochs = max(1, alternate_every_n_epochs)
        # number of image encoders in the SumSAM module
        self.num_encoders = len(self.model.image_encoders)
        # set initial encoder to train (0 by default)
        self._set_encoder_training(0)

    def forward(self, pixel_values, input_boxes, slopes=None, meteo_data=None):
        return self.model(pixel_values=pixel_values, input_boxes=input_boxes)

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
    
    def _set_encoder_training(self, encoder_idx: int):
        """Enable gradients only for the chosen encoder index, disable for others."""
        encoder_idx = int(encoder_idx) % max(1, self.num_encoders)
        
        for i, encoder in enumerate(self.model.image_encoders):
            requires = (i == encoder_idx)
            # Use named_parameters() to access both name and parameter
            for name, param in encoder.named_parameters():
                if "Adapter" in name:
                    param.requires_grad = requires
        
        print(f"Encoder training set -> encoder {encoder_idx} trainable, others frozen.")

    def on_train_epoch_start(self) -> None:
        """Called at start of each training epoch: choose which encoder to train."""
        encoder_idx = (self.current_epoch // self.alternate_every_n_epochs) % max(1, self.num_encoders)
        self._set_encoder_training(encoder_idx)
    
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
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        #optimizer = torch.optim.SGD(
        #    filter(lambda p: p.requires_grad, self.parameters()),
        #    lr=self.learning_rate,
        #    momentum=0.9,      # Added momentum
        #    weight_decay=1e-4  # Optional weight decay
        #)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=20)
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
    ground_truth_mask = np.array(item["label"])
    VH0 = np.array(item["VH0"])
    VH1 = np.array(item["VH1"])
    VV0 = np.array(item["VV0"])
    VV1 = np.array(item["VV1"])
    dem = np.array(item["dem"])
    slope = np.array(item["slope"])
    scale_factor = 2  # typically 2 when target is 1024

    
    image1 = np.stack([VV0, VV1, dem], axis=1)
    image2 = np.stack([VH0, VH1, slope], axis=1)

    ground_truth_mask = torch.from_numpy(ground_truth_mask).float()
    image_tensor1 = torch.from_numpy(image1).float()  # Convert to tensor and change to (C, H, W)
    image_tensor2 = torch.from_numpy(image2).float()  # Convert to tensor and change to (C, H, W)

    bounding_boxes = []

    inputs = {
            "pixel_values": [image_tensor1, image_tensor2],
            "ground_truth_mask": ground_truth_mask,
        }
    
    if self.augment:
        # Stack the list into a single tensor before augmentation
        inputs["pixel_values"] = torch.stack(inputs["pixel_values"], dim=0)
        inputs = RandomAffine()(inputs)
        inputs = RandomFlip()(inputs)
        inputs = RandomNoise()(inputs)
        # After all augmentations, de-stack pixel_values
        inputs["pixel_values"] = list(torch.unbind(inputs["pixel_values"], dim=0))


    if self.test:
        prompt = item["box"]
        for box in prompt:
            x, y, w, h = box
            bounding_boxes.append([[x, y, x + w, y + h]])
    else:
        for mask in inputs["ground_truth_mask"].numpy():
            rand = np.random.rand()
            if rand < 0.8 or not self.augment:
                boxes = create_bounding_boxes(mask, augment=self.augment)
            elif rand < 0.9:
                boxes = create_bounding_boxes(mask, augment=self.augment, increase_by=100)
            else:
                #boxes = create_bounding_boxes(mask, augment=self.augment, increase_by=250)
                boxes = np.array([[0, 0, 512, 512]])

            if len(boxes) == 0:
                #print("No boxes found, using full image")
                #boxes = np.array([[0, 0, 512, 512]])
                # Instead of using the full image box, generate a random box:
                # Define a minimum size for the random box, e.g., 50 pixels.
                min_size = 50
                max_size = 250
                # Random width and height between min_size and max_size.
                w_random = np.random.randint(min_size, max_size)
                h_random = np.random.randint(min_size, max_size)
                # Ensure the box is within boundaries.
                x_random = np.random.randint(0, 512 - w_random)
                y_random = np.random.randint(0, 512 - h_random)
                boxes = np.array([[x_random, y_random, w_random, h_random]])
            else:
                boxes = np.array(boxes)
            #transform boxes to (x1, y1, x2, y2) from (x, y, w, h)
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 0] + boxes[:, 2]
            y2 = boxes[:, 1] + boxes[:, 3]
            # Combine into final boxes in (x1, y1, x2, y2) format
            boxes = np.stack([x1, y1, x2, y2], axis=1)

            bounding_boxes.append(boxes)

    for i in range(len(bounding_boxes)):
        bounding_boxes[i] = torch.tensor(bounding_boxes[i], dtype=torch.float)

    inputs["input_boxes"] = [box * scale_factor for box in bounding_boxes]

    inputs["pixel_values"] = [torch.nn.functional.interpolate(img, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False) for img in inputs["pixel_values"]]

    return inputs