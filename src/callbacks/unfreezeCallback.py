import pytorch_lightning as pl
from sympy import python
import torch

class UnfreezeModelCallback(pl.Callback):
    def __init__(self, unfreeze_epoch: int):
        """
        Args:
            unfreeze_epoch (int): The epoch after which to unfreeze the model.
        """
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    #def on_train_epoch_end(self, trainer, pl_module):
    #    # Check if the current epoch is the one to unfreeze the model
    #    if trainer.current_epoch == self.unfreeze_epoch:
    #        print(f"Unfreezing adapter parts at epoch {trainer.current_epoch}")
    #        # Iterate over model parameters and unfreeze those matching your criteria.
    #        for name, param in pl_module.named_parameters():
    #            if ("Adapter" in name or "mask_decoder" in name) and not param.requires_grad:
    #                param.requires_grad = True
    #                print(f"Unfroze {name}")
#
    #        new_params = [
    #            p for n, p in pl_module.named_parameters() 
    #            if p.requires_grad and not any(p is q for group in trainer.optimizers[0].param_groups for q in group['params'])
    #        ]
#
    #        if new_params:
    #            trainer.optimizers[0].add_param_group({"params": new_params})

    def on_train_epoch_end(self, trainer, pl_module):   
        if trainer.current_epoch == self.unfreeze_epoch:       
            print(f"Unfreezing adapter parts at epoch {trainer.current_epoch}")      
            # Unfreeze chosen parameters           
            for name, param in pl_module.named_parameters():
                if ("Adapter" in name or "mask_decoder" in name) and not param.requires_grad:
                    param.requires_grad = True
                    print(f"Unfroze {name}")
            # Reinitialize the optimizer with all parameters that require gradients           
            new_params = [p for p in pl_module.parameters() if p.requires_grad]      
            # Create a new optimizer (example with AdamW, be sure to use your own settings)           
            new_optim = torch.optim.AdamW(new_params, lr=pl_module.learning_rate)    
            trainer.optimizers[0].param_groups = new_optim.param_groups