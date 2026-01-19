import pytorch_lightning as pl
import numpy as np
import torch
import matplotlib.pyplot as plt
from io import BytesIO

class ImageLoggingCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=5):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            return
        
        # Ensure that validation dataloaders exist
        if not hasattr(trainer, "val_dataloaders") or not trainer.val_dataloaders:
            print("No validation dataloaders found, skipping image logging.")
            return

        # Get one batch from the validation dataloader.
        # Note: To log a "before augmentation" image, you must either save that in your dataset,
        # or use a separate dataloader for non-augmented examples.
        val_loader = trainer.val_dataloaders
        batch = next(iter(val_loader))
        
        # Assume the batch already contains augmented images.
        # If you want before augmentation, modify your dataset to include a copy.
        input_img = batch["pixel_values"][0].cpu()  # as tensor
        ground_truth = batch["ground_truth_mask"][0].cpu()

        # Run prediction.
        pl_module.eval()
        with torch.no_grad():
            pred = pl_module(batch["pixel_values"].to(pl_module.device))
            # Resize prediction to input resolution if needed:
            pred = torch.nn.functional.interpolate(pred, size=input_img.shape[-2:], mode='bilinear', align_corners=False)
        pl_module.train()
        pred_img = pred[0].cpu()

        # Utility to prepare images for display.
        def prepare_img(img, is_mask=False):
            img = img.numpy()
            # If image is channel-first (C,H,W) with C = 1 or 3, convert to HWC.
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1,2,0))
            if not is_mask:
                # Normalize to 0-255.
                img = (img - img.min()) / (img.max() - img.min() + 1e-5)
                img = (img * 255).astype(np.uint8)
            else:
                # If mask is RGB, convert to grayscale.
                if img.ndim == 3 and img.shape[-1] == 3:
                    # Simple averaging to convert to grayscale.
                    img = np.mean(img, axis=2).astype(np.uint8)
            return img

        input_vis = prepare_img(input_img)
        gt_vis = prepare_img(ground_truth, is_mask=True)
        pred_vis = prepare_img(pred_img)

        # Create a figure with three subplots.
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(input_vis)
        axs[0].set_title("Input (augmented)")
        axs[1].imshow(gt_vis, cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[2].imshow(pred_vis)
        axs[2].set_title("Prediction")
        for ax in axs:
            ax.axis('off')

        # Save figure to a buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)

        # Log image using Neptune (or via trainer.logger).
        if trainer.logger is not None:
            # Neptune logger exposes an 'experiment' attribute.
            trainer.logger.experiment.log_image(f"val_samples/epoch_{epoch}", buf)