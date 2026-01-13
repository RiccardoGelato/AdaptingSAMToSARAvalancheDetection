import torch
from datasets import Dataset
import numpy as np
from model.imageAugmentations import RandomRotation90, RandomFlip, RandomMasking, RandomNoise, RandomAffine, process_meteo
from dataprocessing.rcsHandlingFunctions import _rescale
from dataprocessing.creationOfDataframe import create_bounding_boxes
from model.inputTypes import InputTypes


class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor, do_normalize = False, do_rescale = False, augment = True, test_meteo = False):
    self.dataset = dataset
    self.processor = processor
    self.normalize = do_normalize
    self.rescale = do_rescale
    self.augment = augment
    self.test_meteo = test_meteo

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = np.array(item["image"])
    ground_truth_mask = np.array(item["label"])
    slope = np.array(item["slope"]) if "slope" in item else None
    #meteo = item["met"] if "met" in item else None
    meteo = None

    # get bounding box prompt
    # tried -5 5, -5 15, -5 10
    prompt = item["box"]
    xchange = np.random.randint(-5, 5) if self.augment else 0
    ychange = np.random.randint(-5, 5) if self.augment else 0
    wchange = np.random.randint(-5, 5) if self.augment else 0
    hchange = np.random.randint(-5, 5) if self.augment else 0
    input_boxes = []
    for box in prompt: 
      x, y, w, h = box
      input_boxes.append([[max(0, x - xchange), max(0, y - ychange), min(512, x + w + wchange), min(512, y + h + hchange)]])

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=input_boxes,do_normalize = self.normalize, return_tensors="pt", do_rescale = self.rescale)

    # remove batch dimension which the processor adds by default
    inputs = {k:v for k,v in inputs.items()}

    ground_truth_mask = torch.from_numpy(ground_truth_mask)
    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask
    if slope is not None and meteo is not None and self.augment:
        rand = np.random.rand()
        if rand < 0.5:
            slope = torch.from_numpy(slope).float()
            # Add channel dimension -> (B,1, H, W)
            slope = slope.unsqueeze(1)
            inputs["slope"] = slope
            meteo = process_meteo(meteo)
            meteo = torch.from_numpy(meteo).float()
            inputs["meteo"] = meteo
    elif self.test_meteo :
        if slope is not None and meteo is not None:
            #print("Using slope and meteo data")
            slope = torch.from_numpy(slope).float()
            # Add channel dimension -> (B,1, H, W)
            slope = slope.unsqueeze(1)
            inputs["slope"] = slope
            meteo = process_meteo(meteo)
            meteo = torch.from_numpy(meteo).float()
            inputs["meteo"] = meteo
            


    if self.augment:
        inputs = RandomRotation90()(inputs)
        inputs = RandomFlip()(inputs)
        inputs = RandomMasking()(inputs)
        #inputs = RandomNoise()(inputs)
        
    inputs["input_boxes"] = inputs["input_boxes"].squeeze(1)

    return inputs

class SAMDataset3(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor, do_normalize = False, do_rescale = False, augment = True, target_size=1024, type = InputTypes.Normal, test = False):
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

    if self.type == InputTypes.VV:
        image = np.stack([VV0, VV1, dem], axis=1)
    elif self.type == InputTypes.VH:
        image = np.stack([VH0, VH1, slope], axis=1)
        #image = np.stack([VH0, VH1, dem], axis=1)
    elif self.type == InputTypes.Normal:
        a = _rescale(VH1 - VH0, 0, .25)

        b = _rescale(VV1 - VV0, 0, .25)

        w = _rescale(a - b, 0, 1)

        r = w*VH0 + (1 - w)*VV0

        g = w*VH1 + (1 - w)*VV1

        image = np.stack([r, g, dem], axis=1)
    elif self.type == InputTypes.Difference:
        image = np.stack([VV1-VV0, VH1-VH0, dem], axis=1)
    else:
        # Combine all channels to create image
        image = np.stack([VH0, VH1, VV0, VV1, dem, slope], axis=1)

    ground_truth_mask = torch.from_numpy(ground_truth_mask).float()
    image_tensor = torch.from_numpy(image).float()  # Convert to tensor and change to (C, H, W)
    
    bounding_boxes = []

    inputs = {
            "pixel_values": image_tensor,
            "ground_truth_mask": ground_truth_mask,
        }
    
    if self.augment:
        inputs = RandomAffine()(inputs)
        inputs = RandomFlip()(inputs)
        inputs = RandomNoise()(inputs)

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
            #xchange = np.random.randint(-5, 5) if self.augment else 0
            #ychange = np.random.randint(-5, 5) if self.augment else 0
            #wchange = np.random.randint(-5, 5) if self.augment else 0
            #hchange = np.random.randint(-5, 5) if self.augment else 0
            ## Assume boxes is originally in (x, y, w, h) format.
            #x_orig = boxes[:, 0].copy()
            #y_orig = boxes[:, 1].copy()
            #w_orig = boxes[:, 2].copy()
            #h_orig = boxes[:, 3].copy()
#
            ## Apply jitter to the top-left coordinates
            #x1 = np.clip(x_orig - xchange, 0, 512)
            #y1 = np.clip(y_orig - ychange, 0, 512)
#
            ## Compute bottom-right coordinates using the original width/height plus jitter
            #x2 = np.clip(x_orig + w_orig + wchange, 0, 512)
            #y2 = np.clip(y_orig + h_orig + hchange, 0, 512)
#
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
    inputs["pixel_values"] = torch.nn.functional.interpolate(inputs["pixel_values"], size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)

    return inputs