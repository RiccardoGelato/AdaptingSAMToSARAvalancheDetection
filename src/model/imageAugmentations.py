import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import numpy as np


class RandomRotation90:
    def __call__(self, inputs, mode = 'standard'):
        img = inputs["pixel_values"]
        mask = inputs["ground_truth_mask"]
        boxes = inputs['input_boxes'] if 'input_boxes' in inputs else None
        slope = inputs['slope'] if 'slope' in inputs else None
        if mode == 'standard':
            angle = np.random.choice([0, 90, 180, 270])
        else:
            angle = np.random.uniform(0, 360)
        img = v2.functional.rotate(img, angle)
        mask = v2.functional.rotate(mask, angle)
        if boxes is not None:
            boxes = self.rotate_boxes(boxes, angle, img.shape[-2:])
        if slope is not None:
            slope = v2.functional.rotate(slope, angle)
        inputs["pixel_values"] = img
        inputs["ground_truth_mask"] = mask
        if boxes is not None:
            inputs['input_boxes'] = boxes
        if slope is not None:
            inputs['slope'] = slope
        return inputs

    def rotate_boxes(self, boxes, angle, img_size):
        w, h = img_size
        if angle == 90:
            return torch.tensor([[y1, w - x2, y2, w - x1] for x1, y1, x2, y2 in boxes.view(-1, 4)]).view(boxes.shape)
        elif angle == 180:
            return torch.tensor([[w - x2, h - y2, w - x1, h - y1] for x1, y1, x2, y2 in boxes.view(-1, 4)]).view(boxes.shape)
        elif angle == 270:
            return torch.tensor([[h - y2, x1, h - y1, x2] for x1, y1, x2, y2 in boxes.view(-1, 4)]).view(boxes.shape)
        return boxes
    
class RandomFlip:
    def __init__(self, p=0.66):
        self.p = p

    def __call__(self, inputs):
        img = inputs["pixel_values"]
        mask = inputs["ground_truth_mask"]
        boxes = inputs['input_boxes'] if 'input_boxes' in inputs else None
        slope = inputs['slope'] if 'slope' in inputs else None

        if np.random.random() < self.p:
            if np.random.choice([True, False]):
                # Apply horizontal flip
                img = v2.functional.hflip(img)
                mask = v2.functional.hflip(mask)
                if boxes is not None:
                    boxes = self.flip_boxes_horizontally(boxes, img.shape[-1])
                if slope is not None:
                    slope = v2.functional.hflip(slope)
            else:
                # Apply vertical flip
                img = v2.functional.vflip(img)
                mask = v2.functional.vflip(mask)
                if boxes is not None:
                    boxes = self.flip_boxes_vertically(boxes, img.shape[-2])
                if slope is not None:
                    slope = v2.functional.vflip(slope)

        inputs["pixel_values"] = img
        inputs["ground_truth_mask"] = mask
        if boxes is not None:
            inputs['input_boxes'] = boxes
        if slope is not None:
            inputs['slope'] = slope
        return inputs

    def flip_boxes_horizontally(self, boxes, img_width):
        return torch.tensor([[img_width - x2, y1, img_width - x1, y2] for x1, y1, x2, y2 in boxes.view(-1, 4)]).view(boxes.shape)

    def flip_boxes_vertically(self, boxes, img_height):
        return torch.tensor([[x1, img_height - y2, x2, img_height - y1] for x1, y1, x2, y2 in boxes.view(-1, 4)]).view(boxes.shape)
    
class RandomMasking:
    def __init__(self, max_size=256, mask_value=0, max_masks=2, ):
        self.max_size = max_size
        self.mask_value = mask_value
        self.max_masks = max_masks

    def __call__(self, inputs):
        img = inputs["pixel_values"]
        boxes = inputs['input_boxes']

        # Get image dimensions
        _,_, h, w = img.shape

        # Randomly choose the number of masks to apply
        num_masks = np.random.randint(0, self.max_masks + 1)

        for _ in range(num_masks):
            # Randomly choose the size of the mask
            mask_size = np.random.randint(1, self.max_size + 1)

            # Randomly choose the top-left corner of the mask
            top = np.random.randint(0, h - mask_size)
            left = np.random.randint(0, w - mask_size)

            # Check if the mask overlaps with any of the boxes
            for index, box in enumerate(boxes):
                overlaps = False
                x1, y1, x2, y2 = box[0]
                if not (left + mask_size < x1 or left > x2 or top + mask_size < y1 or top > y2):
                    overlaps = True
                    
                if not overlaps:
                    # Apply the mask to the image
                    img[index, :, top:top + mask_size, left:left + mask_size] = self.mask_value

        inputs["pixel_values"] = img
        return inputs
    
class RandomNoise:
    def __init__(self, noise_factor=0.01):
        self.noise_factor = noise_factor

    def __call__(self, inputs):
        img = inputs["pixel_values"]

        inputs["pixel_values"] = v2.GaussianNoise(clip=False, sigma = self.noise_factor)(img)

        return inputs
    
def process_meteo(meteo, expected_length=242):
    """
    Processes meteo data ensuring that each element has the same length.
    Each time series is padded with zeros (or truncated) to have `expected_length` steps.
    Returns an array of shape (N, expected_length, feature_dim) where N = len(meteo).
    """
    processed = []
    for m in meteo:
        m = np.array(m)  # convert each element to array
        # Replace NaN, positive inf, and negative inf values with 0.0
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        current_length = m.shape[0]
        if current_length < expected_length:
            # determine pad for each dimension; assume m is 1D or 2D (T, F)
            pad_width = ((0, expected_length - current_length),) + ((0, 0),) * (m.ndim - 1)
            m = np.pad(m, pad_width=pad_width, mode='constant', constant_values=0)
        elif current_length > expected_length:
            m = m[:expected_length]
        processed.append(m)
    # Stack into a single array if possible
    return np.stack(processed, axis=0)

class RandomAffine:
    def __init__(self, max_translation=50):
        """
        Args:
            max_translation (int): Maximum translation (in px) in both x and y directions.
        """
        self.max_translation = max_translation

    def __call__(self, inputs):

        img = inputs["pixel_values"]
        mask = inputs["ground_truth_mask"]

        # Choose random rotation angle
        angle = np.random.uniform(0, 360)
        #angle = int(np.random.choice([0, 90, 180, 270]))
        # Choose random translation offsets (dx, dy)
        tx = np.random.randint(-self.max_translation, self.max_translation)
        ty = np.random.randint(-self.max_translation, self.max_translation)
        # Use affine transform with angle rotation, scale=1, no shear.
        img = v2.functional.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=0.0)
        mask = v2.functional.affine(mask, angle=angle, translate=(tx, ty), scale=1.0, shear=0.0)

        inputs["pixel_values"] = img
        inputs["ground_truth_mask"] = mask
        return inputs