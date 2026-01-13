import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


# Function to find bounding boxes for each group of disconnected white pixels
def find_bounding_boxes(mask):
    # Ensure the mask is an 8-bit image.
    if mask.dtype != "uint8":
        # If mask values are in range 0-1, scale them by 255
        if mask.max() <= 1:
            mask_uint8 = (mask * 255).astype('uint8')
        else:
            mask_uint8 = mask.astype('uint8')
    else:
        mask_uint8 = mask

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute bounding boxes for each contour and convert (x, y, w, h) to (x_min, y_min, x_max, y_max)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append([x, y, x + w, y + h])
    return bounding_boxes

def calculate_iou(mask1, mask2):

    # Ensure the masks are PyTorch tensors
    if isinstance(mask1, np.ndarray):
        mask1 = torch.tensor(mask1)
    if isinstance(mask2, np.ndarray):
        mask2 = torch.tensor(mask2)
        
    # Ensure the masks are binary
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    
    # Calculate the intersection and union
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    
    # Compute the IoU
    iou = torch.sum(intersection).float() / torch.sum(union).float()
    
    return iou, intersection, union

def compute_error_percentages(results, threshold=0.3, find_bounding_boxes_fn=find_bounding_boxes):
    """
    Computes false negatives and false positives percentages for each result.
    
    Parameters:
        results (list): List of dictionaries. Each dictionary must contain:
                        - 'mask': ground truth binary mask (numpy array)
                        - 'calculated_mask': predicted binary mask (numpy array)
        threshold (float): Overlap threshold to decide if a bounding box was predicted.
        find_bounding_boxes_fn (function): A function to compute bounding boxes given a mask.
                                             If None, the function must be defined elsewhere.
                                             
    Returns:
        list: The updated results list with the following keys added:
              - 'false_negatives': List of bounding boxes not predicted.
              - 'false_positives': List of bounding boxes falsely predicted.
              - 'percentage_false_negatives': Ratio of false negatives to total GT boxes.
              - 'percentage_false_positives': Ratio of false positives to total predicted boxes.
    """
    if find_bounding_boxes_fn is None:
        raise ValueError("A function for finding bounding boxes must be provided.")

    for res in results:
        false_positives = []
        false_negatives = []
        pred_mask = res['calculated_mask']
        true_mask = res['mask']

        original_bbox = find_bounding_boxes_fn(true_mask)
        pred_bbox = find_bounding_boxes_fn(pred_mask)

        # Process false negatives: ground truth boxes with no matching predicted box.
        for bbox in original_bbox:
            predicted = False
            for pred in pred_bbox:
                # Create local masks for current predicted bbox and GT bbox.
                pred_mask_local = np.zeros_like(pred_mask)
                pred_mask_local[pred[1]:pred[3], pred[0]:pred[2]] = pred_mask[pred[1]:pred[3], pred[0]:pred[2]]

                original_mask_local = np.zeros_like(true_mask)
                original_mask_local[bbox[1]:bbox[3], bbox[0]:bbox[2]] = true_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # Calculate overlap area.
                overlap_area = np.sum(np.logical_and(pred_mask_local, original_mask_local))
                original_area = np.sum(original_mask_local)
                overlap = overlap_area / original_area if original_area > 0 else 0

                if overlap > threshold:
                    predicted = True
                    break
            if not predicted:
                false_negatives.append(bbox)

        # Process false positives: predicted boxes that do not have corresponding GT boxes.
        for bbox in pred_bbox:
            original = False
            for originalbbox in original_bbox:
                pred_mask_local = np.zeros_like(pred_mask)
                pred_mask_local[bbox[1]:bbox[3], bbox[0]:bbox[2]] = pred_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                original_mask_local = np.zeros_like(true_mask)
                original_mask_local[originalbbox[1]:originalbbox[3], originalbbox[0]:originalbbox[2]] = true_mask[originalbbox[1]:originalbbox[3], originalbbox[0]:originalbbox[2]]

                overlap_area = np.sum(np.logical_and(pred_mask_local, original_mask_local))
                pred_area = np.sum(pred_mask_local)
                overlap = overlap_area / pred_area if pred_area > 0 else 0

                if overlap > threshold:
                    original = True
                    break
            if not original:
                false_positives.append(bbox)

        # Update the result dictionary with computed metrics.
        res['false_negatives'] = false_negatives
        res['false_positives'] = false_positives
        res['percentage_false_negatives'] = len(false_negatives) / len(original_bbox) if len(original_bbox) > 0 else 0
        res['percentage_false_positives'] = len(false_positives) / len(pred_bbox) if len(pred_bbox) > 0 else 0

    return results

def plot_iou_statistics(df, iou_column='iou', model_name='Baseline'):
    """
    Extracts IoU values from the DataFrame, calculates the average IoU,
    and plots a bar chart, a line chart, and a histogram of the IoU scores.

    Parameters:
        df (pandas.DataFrame): DataFrame containing IoU scores.
        iou_column (str): Column name in df that contains the IoU values (default: 'iou').
    """
    # Extract IoU values from the DataFrame
    iou_values = df[iou_column].tolist()
    
    # Calculate the average IoU
    if len(iou_values) == 0:
        average_iou = 0
    else:
        average_iou = sum(iou_values) / len(iou_values)
    
    ## Bar chart
    #plt.figure(figsize=(10, 6))
    #plt.bar(range(len(iou_values)), iou_values, color='blue')
    #plt.xlabel('Sample Index')
    #plt.ylabel('IoU')
    #plt.title(f'IoU Values for Different Samples ({model_name})')
    #plt.text(0.5, 0.95, f'Average IoU: {average_iou:.4f}', ha='center', va='center',
    #         transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    #plt.show()
    
    ## Line plot
    #plt.figure(figsize=(10, 6))
    #plt.plot(range(len(iou_values)), iou_values, marker='o', linestyle='-', color='blue')
    #plt.xlabel('Sample Index')
    #plt.ylabel('IoU')
    #plt.title(f'IoU Values for Different Samples ({model_name})')
    #plt.text(0.5, 0.95, f'Average IoU: {average_iou:.4f}', ha='center', va='center',
    #         transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    #plt.show()
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(iou_values, bins=20, color='blue', alpha=0.7)
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of IoU Scores (Average IoU: {average_iou:.4f}, {model_name})')
    plt.show()

def generate_results(test_dataset, ground_truth_masks, predicted_masks, bounding_boxes, calculate_iou = calculate_iou):
    """
    Generates a list of results with IoU evaluation for each sample in the test dataset.

    Parameters:
        test_dataset (Dataset): The test dataset.
        ground_truth_masks (numpy.ndarray): Array of ground truth masks.
        predicted_masks (numpy.ndarray): Array of predicted masks.
        bounding_boxes (numpy.ndarray): Array of bounding boxes corresponding to each sample.
        calculate_iou (function): Function to calculate IoU that returns (iou, intersection, union).

    Returns:
        list: A list of dictionaries, each containing:
              - 'mask': ground truth mask,
              - 'calculated_mask': predicted binary mask,
              - 'intersection': intersection mask (numpy array),
              - 'union': union mask (numpy array),
              - 'iou': computed IoU (numpy array),
              - 'empty': always False,
              - 'bbox': bounding box for the sample,
              - 'image_id': index of the sample.
    """
    results = []
    for index in range(len(test_dataset)):
        mask = ground_truth_masks[index]
        sam_seg = predicted_masks[index]

        # Convert the predicted mask to a probability map and then to a hard mask
        sam_seg_prob = torch.sigmoid(torch.tensor(sam_seg))
        sam_seg_prob = sam_seg_prob.cpu().numpy().squeeze()
        sam_seg_hard = (sam_seg_prob > 0.5).astype(np.uint8)

        # Calculate IoU and associated masks
        iou, intersection, union = calculate_iou(mask, sam_seg_hard)

        results.append({
            'mask': mask,
            'calculated_mask': sam_seg_hard,
            'intersection': intersection.cpu().squeeze().numpy(),
            'union': union.cpu().squeeze().numpy(),
            'iou': iou.cpu().numpy(),
            'empty': False,
            'bbox': bounding_boxes[index],
            'image_id': index,
        })
    return results

def plot_iou_vs_area_ratio(df, model_name='Baseline'):
    """
    Computes the bounding box area and the ratio of mask area over bbox area,
    and then produces two scatter plots of IoU against the area ratio.
    
    Parameters:
        df (pandas.DataFrame): DataFrame that must include the following columns:
            - 'bbox': list or array with bounding box info in the format [x1, y1, x2, y2]
            - 'mask_area': numeric mask area values
            - 'iou': IoU values.
    """
    # Calculate the area of each bounding box (assumes bbox format is (x1, y1, x2, y2))
    df['bbox_area'] = df['bbox'].apply(
    lambda b: (b[2] - b[0]) * (b[3] - b[1]) / 4 if (b is not None and len(b) == 4) else None
    )
    
    # Compute the ratio of bounding box area to mask area
    df['area_ratio'] = df['mask_area'] / df['bbox_area']
    
    # Drop rows with missing values in area_ratio or IoU to ensure clean plotting
    df_plot = df.dropna(subset=['area_ratio', 'iou'])
    
    # Scatter plot of IoU vs. area ratio (linear scale)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_plot['area_ratio'], df_plot['iou'], marker='o', color='blue')
    plt.xlabel('Bounding Box Area / Mask Area')
    plt.ylabel('IoU')
    plt.title(f'IoU vs. Area Ratio ({model_name})')
    plt.grid(True)
    plt.show()
    
    # Scatter plot of IoU vs. area ratio with logarithmic x-axis
    plt.figure(figsize=(10, 6))
    plt.scatter(df_plot['area_ratio'], df_plot['iou'], marker='o', color='blue')
    plt.xscale('log')
    plt.xlabel('Bounding Box Area / Mask Area (log scale)')
    plt.ylabel('IoU')
    plt.title(f'IoU vs. Area Ratio (log scale) ({model_name})')
    plt.grid(True)
    plt.show()

def plot_iou_vs_num_avalanches(df, model_name='Baseline'):
    """
    Computes the number of disconnected regions (avalanches) in each mask,
    adds a new column 'num_avalanches' to the DataFrame, and plots a scatter plot
    of IoU versus the number of avalanches.

    Parameters:
        df (pandas.DataFrame): DataFrame that must contain:
            - 'mask': the mask array (binary or grayscale)
            - 'iou': IoU values for each sample.
    """
    # Define function to count avalanches in a mask.
    def count_avalanches(mask):
        # Ensure the mask is binary: any non-zero pixel is part of an avalanche.
        binary_mask = (mask > 0).astype(np.uint8)
        # cv2.connectedComponents returns (num_labels, labels); subtract 1 for the background.
        num_labels, _ = cv2.connectedComponents(binary_mask)
        return num_labels - 1  # subtract one for the background

    # Compute the number of avalanches for each mask.
    df['num_avalanches'] = df['mask'].apply(lambda m: count_avalanches(m) if m is not None else None)
    
    # Drop rows with missing values in 'num_avalanches' or 'iou'
    df_plot = df.dropna(subset=['num_avalanches', 'iou'])
    
    # Plot IoU vs. number of avalanches.
    plt.figure(figsize=(10, 6))
    plt.scatter(df_plot['num_avalanches'], df_plot['iou'], marker='o', color='blue')
    plt.xlabel('Number of Avalanches in Mask')
    plt.ylabel('IoU')
    plt.title(f'IoU vs. Number of Avalanches ({model_name})')
    plt.grid(True)
    plt.show()

def plot_iou_for_mask_area(df, mask_area_threshold=100, model_name='Baseline'):
    """
    Filters the DataFrame to include only rows with non-missing 'mask_area' and 'iou'
    and with mask_area greater than the provided threshold. Then, it extracts the IoU 
    values, calculates the average IoU, and produces both a bar chart and a line plot.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing at least the 'mask_area' and 'iou' columns.
        mask_area_threshold (int, optional): Minimum mask area to include. Default is 100.
    """
    if 'mask_area' not in df.columns:
        # Calculate the area of each mask
        df['mask_area'] = df['mask'].apply(lambda mask: np.sum(mask))
    # Filter out rows where mask_area or IoU are missing and select only those with mask_area above the threshold.
    df_filtered_threshold = df.dropna(subset=['mask_area', 'iou'])
    df_filtered_threshold = df_filtered_threshold[df_filtered_threshold['mask_area'] > mask_area_threshold]
    
    # Extract IoU values from the filtered DataFrame.
    iou_values = df_filtered_threshold['iou'].tolist()
    
    # Calculate the average IoU (ensure there is at least one value)
    average_iou = sum(iou_values) / len(iou_values) if iou_values else 0

    ## Create a bar chart.
    #plt.figure(figsize=(10, 6))
    #plt.bar(range(len(iou_values)), iou_values, color='blue')
    #plt.xlabel('Sample Index')
    #plt.ylabel('IoU')
    #plt.title(f'IoU Values for Samples with Mask Area > {mask_area_threshold} ({model_name})')
    #plt.text(0.5, 0.95, f'Average IoU: {average_iou:.4f}', ha='center', va='center',
    #         transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    #plt.show()
    #
    ## Create a line plot.
    #plt.figure(figsize=(10, 6))
    #plt.plot(range(len(iou_values)), iou_values, marker='o', linestyle='-', color='blue')
    #plt.xlabel('Sample Index')
    #plt.ylabel('IoU')
    #plt.title(f'IoU Values for Samples with Mask Area > {mask_area_threshold} ({model_name})')
    #plt.text(0.5, 0.95, f'Average IoU: {average_iou:.4f}', ha='center', va='center',
    #         transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    #plt.show()

    # Create a histogram of IoU values.
    plt.figure(figsize=(10, 6))
    plt.hist(iou_values, bins=20, color='blue', alpha=0.7)
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of IoU Scores (Average IoU: {average_iou:.4f}, Mask Area > {mask_area_threshold}, {model_name})')
    plt.show()

def compute_mask_area_iou_correlation(df, calculate_correlation, model_name='Baseline', scale='linear'):
    """
    Converts the 'mask_area' and 'iou' columns of the DataFrame into numeric numpy arrays,
    computes the correlation using the provided calculate_correlation function, prints the result,
    and returns the correlation and p-value.

    Parameters:
        df (pandas.DataFrame): DataFrame that contains 'mask_area' and 'iou' columns.
        calculate_correlation (function): A function that takes two numpy arrays and returns (corr, p_value).
    
    Returns:
        tuple: (corr, p_value)
    """
    if scale == 'log':
        mask_area_numeric = np.log(np.asarray(df['mask_area'].values, dtype=np.float32) + 1e-8)
    else:
        mask_area_numeric = np.asarray(df['mask_area'].values, dtype=np.float32)

    iou_numeric = np.asarray(df['iou'].values, dtype=np.float32)

    corr, p_value = calculate_correlation(mask_area_numeric, iou_numeric)

    print(f"Correlation between mask area and IoU ({model_name}, scale={scale}): {corr}, p-value: {p_value}")
    return corr, p_value


def compute_num_avalanche_iou_correlation(df, calculate_correlation, model_name='Baseline'):
    """
    Converts the 'num_avalanches' and 'iou' columns of the DataFrame into numeric numpy arrays,
    computes their correlation using the provided calculate_correlation function,
    prints and returns the correlation coefficient and p-value.

    Parameters:
        df (pandas.DataFrame): DataFrame containing at least the 'num_avalanches' and 'iou' columns.
        calculate_correlation (function): Function that takes two numpy arrays and returns (corr, p_value).

    Returns:
        tuple: (corr, p_value)
    """
    num_avalanches_numeric = np.asarray(df['num_avalanches'].values, dtype=np.float32)
    iou_numeric = np.asarray(df['iou'].values, dtype=np.float32)

    corr, p_value = calculate_correlation(num_avalanches_numeric, iou_numeric)
    print(f"Correlation between number of avalanches and IoU ({model_name}): {corr}, p-value: {p_value}")
    return corr, p_value

def compute_area_ratio_iou_correlation(df, calculate_correlation, model_name='Baseline', scale='linear'):
    """
    Converts the 'area_ratio' and 'iou' columns of the DataFrame into numeric numpy arrays,
    computes their correlation using the provided calculate_correlation function,
    prints and returns the correlation coefficient and p-value.

    Parameters:
        df (pandas.DataFrame): DataFrame that contains 'area_ratio' and 'iou' columns.
        calculate_correlation (function): A function that takes two numpy arrays and returns (corr, p_value).

    Returns:
        tuple: (corr, p_value)
    """
    if scale == 'log':
        area_ratio_numeric = np.log(np.asarray(df['area_ratio'].values, dtype=np.float32) + 1e-8)
    else:
        area_ratio_numeric = np.asarray(df['area_ratio'].values, dtype=np.float32)
    iou_numeric = np.asarray(df['iou'].values, dtype=np.float32)
    
    corr, p_value = calculate_correlation(area_ratio_numeric, iou_numeric)
    print(f"Correlation between area ratio and IoU ({model_name}, scale={scale}): {corr}, p-value: {p_value}")
    return corr, p_value


def plot_mask_area_vs_iou(df, area_threshold=1000, model_name='Baseline'):
    """
    Computes the area of each mask from the 'mask' column, filters the DataFrame by area_threshold,
    and produces two scatter plots comparing Mask Area vs. IoU. The first plot shows a standard scale,
    and the second is plotted with a logarithmic scale on the x-axis.
    
    Parameters:
        df (pandas.DataFrame): DataFrame that must contain at least 'mask' and 'iou' columns.
        area_threshold (int, optional): Maximum mask area to include in the standard scale plot.
            Rows with a mask area above this threshold will be omitted. Default is 1000.
    """
    # Calculate the area of each mask
    df['mask_area'] = df['mask'].apply(lambda mask: np.sum(mask))
    
    # Filter the DataFrame for the standard scale plot
    df_filtered = df[df['mask_area'] <= area_threshold]
    
    # Plot with standard scale
    plt.figure(figsize=(10, 6))
    plt.scatter(df_filtered['mask_area'], df_filtered['iou'], marker='o', color='blue')
    plt.xlabel('Mask Area')
    plt.ylabel('IoU')
    plt.title(f'IoU vs. Mask Area (Standard Scale, {model_name})')
    plt.grid(True)
    plt.show()
    
    # Plot with logarithmic scale for the mask area axis
    plt.figure(figsize=(10, 6))
    plt.scatter(df['mask_area'], df['iou'], marker='o', color='blue')
    plt.xscale('log')
    plt.xlabel('Mask Area (log scale)')
    plt.ylabel('IoU')
    plt.title(f'IoU vs. Mask Area (Log Scale, {model_name})')
    plt.grid(True)
    plt.show()

def plot_false_pos_neg_percentages(results, treshold, model_name='Baseline'):
    """
    Plots a bar chart showing the percentage of false negatives and false positives
    for each sample, along with the average percentages.

    Parameters:
        results (list of dict): List containing result dictionaries which must include
                                'percentage_false_negatives' and 'percentage_false_positives'.
        treshold (float): The threshold value used in the analysis (displayed in the title).
    """
    # Extract percentages from the results
    false_negatives_pct = [res.get('percentage_false_negatives', 0) for res in results]
    false_positives_pct = [res.get('percentage_false_positives', 0) for res in results]
    indices = np.arange(len(results))
    
    # Compute average percentages
    avg_false_negatives = np.mean(false_negatives_pct)
    avg_false_positives = np.mean(false_positives_pct)
    
    # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(indices - 0.15, false_negatives_pct, width=0.3, color='red', label='False Negatives')
    plt.bar(indices + 0.15, false_positives_pct, width=0.3, color='blue', label='False Positives')
    
    # Plot average lines
    plt.axhline(avg_false_negatives, color='darkred', linestyle='--', 
                label=f'Avg False Negatives: {avg_false_negatives:.2f}')
    plt.axhline(avg_false_positives, color='darkblue', linestyle='--', 
                label=f'Avg False Positives: {avg_false_positives:.2f}')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Percentage')
    plt.title(f'Percentage of False Negatives and False Positives per Sample ({model_name}, Treshold = {treshold})')
    plt.legend()
    plt.show()

import numpy as np
from scipy.ndimage import label

def extract_individual_masks(binary_mask):
    """
    Extract individual connected component masks from a binary mask.
    
    Parameters:
        binary_mask (numpy array): Binary mask where avalanches are marked as 1
        
    Returns:
        list: List of individual binary masks, one for each connected component
    """
    labeled_mask, num_features = label(binary_mask)
    individual_masks = []
    
    for i in range(1, num_features + 1):
        component_mask = (labeled_mask == i).astype(np.uint8)
        individual_masks.append(component_mask)
    
    return individual_masks


def calculate_mask_iou(mask1, mask2):
    """
    Calculate IoU between two binary masks.
    
    Parameters:
        mask1, mask2 (numpy array): Binary masks
        
    Returns:
        float: IoU score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_precision_recall_iou(results, iou_threshold=0.5, extract_masks_fn=extract_individual_masks):
    """
    Computes precision and recall using IoU-based matching of avalanche instances.
    IoU is calculated on the actual masks, not bounding boxes.
    
    For each image:
    1. Extract individual avalanche masks from ground truth and predictions
    2. Match predicted avalanches to ground truth using mask IoU threshold
    3. Calculate TP, FP, FN for instance-level detection
    4. Compute precision and recall across all images
    
    Parameters:
        results (list): List of dictionaries. Each dictionary must contain:
                        - 'mask': ground truth binary mask (numpy array)
                        - 'calculated_mask': predicted binary mask (numpy array)
        iou_threshold (float): IoU threshold to decide if a detection is a true positive.
        extract_masks_fn (function): A function to extract individual avalanche masks.
                                     Should return list of binary masks.
    
    Returns:
        tuple: (results, summary_metrics)
            - results: Updated list with per-image metrics
            - summary_metrics: Dictionary with overall precision, recall, F1, etc.
    """
    if extract_masks_fn is None:
        raise ValueError("A function for extracting individual masks must be provided.")
    
    # Aggregate counts across all images
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt_avalanches = 0
    total_pred_avalanches = 0
    
    for res in results:
        pred_mask = res['calculated_mask']
        true_mask = res['mask']
        
        # Extract individual avalanche masks
        gt_masks = extract_masks_fn(true_mask)
        pred_masks = extract_masks_fn(pred_mask)
        
        num_gt = len(gt_masks)
        num_pred = len(pred_masks)
        
        total_gt_avalanches += num_gt
        total_pred_avalanches += num_pred
        
        # Track matched ground truth and predicted masks
        matched_gt = set()
        matched_pred = set()
        
        # Store IoU values for analysis
        iou_matrix = np.zeros((num_pred, num_gt))
        
        # For each predicted avalanche, find best matching ground truth
        for pred_idx, pred_single_mask in enumerate(pred_masks):
            best_iou = 0
            best_gt_idx = -1
            
            # Calculate IoU with all ground truth masks
            for gt_idx, gt_single_mask in enumerate(gt_masks):
                if gt_idx in matched_gt:  # Skip already matched GT
                    continue
                
                # Calculate IoU between the actual masks
                iou = calculate_mask_iou(pred_single_mask, gt_single_mask)
                iou_matrix[pred_idx, gt_idx] = iou
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # If best IoU exceeds threshold, mark as match (True Positive)
            if best_iou >= iou_threshold and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
        
        # Calculate TP, FP, FN for this image
        tp = len(matched_pred)  # Successfully matched predictions
        fp = num_pred - tp      # Predicted avalanches with no match
        fn = num_gt - len(matched_gt)  # Ground truth avalanches with no match
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Store per-image results
        false_negative_masks = [gt_masks[i] for i in range(num_gt) if i not in matched_gt]
        false_positive_masks = [pred_masks[i] for i in range(num_pred) if i not in matched_pred]
        
        res['true_positives'] = tp
        res['false_positives'] = fp
        res['false_negatives'] = fn
        res['false_negative_masks'] = false_negative_masks
        res['false_positive_masks'] = false_positive_masks
        res['num_gt_avalanches'] = num_gt
        res['num_pred_avalanches'] = num_pred
        res['iou_matrix'] = iou_matrix  # Store for detailed analysis
        
        # Per-image precision and recall
        res['precision'] = tp / num_pred if num_pred > 0 else 0
        res['recall'] = tp / num_gt if num_gt > 0 else 0
        res['f1_score'] = (2 * res['precision'] * res['recall'] / 
                          (res['precision'] + res['recall'])) if (res['precision'] + res['recall']) > 0 else 0
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    summary_metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_true_positives': total_tp,
        'total_false_positives': total_fp,
        'total_false_negatives': total_fn,
        'total_gt_avalanches': total_gt_avalanches,
        'total_pred_avalanches': total_pred_avalanches,
        'iou_threshold': iou_threshold
    }
    
    return results, summary_metrics


def print_metrics(summary_metrics):
    """Helper function to print the metrics nicely."""
    print("\n" + "="*50)
    print("DETECTION METRICS (Mask IoU-based matching)")
    print("="*50)
    print(f"IoU Threshold: {summary_metrics['iou_threshold']:.2f}")
    print(f"\nTotal Ground Truth Avalanches: {summary_metrics['total_gt_avalanches']}")
    print(f"Total Predicted Avalanches: {summary_metrics['total_pred_avalanches']}")
    print(f"\nTrue Positives (TP): {summary_metrics['total_true_positives']}")
    print(f"False Positives (FP): {summary_metrics['total_false_positives']}")
    print(f"False Negatives (FN): {summary_metrics['total_false_negatives']}")
    print(f"\n{'Metric':<15} {'Value':<10}")
    print("-"*25)
    print(f"{'Precision':<15} {summary_metrics['precision']:.4f}")
    print(f"{'Recall':<15} {summary_metrics['recall']:.4f}")
    print(f"{'F1-Score':<15} {summary_metrics['f1_score']:.4f}")
    print("="*50 + "\n")


## Usage example:
#results, metrics = compute_precision_recall_iou(
#    results=your_results_list,
#    iou_threshold=0.5,
#    extract_masks_fn=extract_individual_masks
#)
#
#print_metrics(metrics)
#
## Access per-image metrics
#for i, res in enumerate(results):
#    print(f"Image {i}: Precision={res['precision']:.3f}, Recall={res['recall']:.3f}, F1={res['f1_score']:.3f}")
#    print(f"  GT avalanches: {res['num_gt_avalanches']}, Pred avalanches: {res['num_pred_avalanches']}")
#    print(f"  TP: {res['true_positives']}, FP: {res['false_positives']}, FN: {res['false_negatives']}")



def calculate_pixel_based_metrics(results):
    """
    Calculate pixel-level precision, recall, and F1.
    Treats all avalanche pixels equally, regardless of instances.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    for res in results:
        pred_mask = res['calculated_mask']
        true_mask = res['mask']
        
        # Flatten to 1D
        pred_flat = pred_mask.flatten()
        true_flat = true_mask.flatten()
        
        # Calculate confusion matrix
        tp = np.sum((pred_flat == 1) & (true_flat == 1))
        fp = np.sum((pred_flat == 1) & (true_flat == 0))
        fn = np.sum((pred_flat == 0) & (true_flat == 1))
        tn = np.sum((pred_flat == 0) & (true_flat == 0))
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        
        # Per-image metrics
        res['pixel_tp'] = tp
        res['pixel_fp'] = fp
        res['pixel_fn'] = fn
        res['pixel_tn'] = tn
        res['pixel_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        res['pixel_recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        res['pixel_f1'] = (2 * res['pixel_precision'] * res['pixel_recall'] / 
                          (res['pixel_precision'] + res['pixel_recall'])) if (res['pixel_precision'] + res['pixel_recall']) > 0 else 0
    
    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_tn': total_tn,
        'total_pixels': total_tp + total_fp + total_fn + total_tn
    }
    
    return results, metrics