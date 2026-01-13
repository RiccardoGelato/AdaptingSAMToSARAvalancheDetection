# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from sam.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, MaskDecoderHQ, ImageEncoderIrViT


def build_sam_vit_h(checkpoint=None, adapt=False, HQ=False, IR=False, selfsup=False, increase_resolution=False, adapt_patch_embed=False):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        adapt=adapt,
        HQ=HQ,
        IR=IR,
        mlp_ratio_adapter=0.25,  # Default value for mlp_ratio in SAM
        dropout_prob=0.0,  # Default value for dropout in SAM
        selfsup=selfsup,  # Flag for self-supervised training
        increase_resolution=increase_resolution,  # Enable resolution increase for SAM
        adapt_patch_embed=adapt_patch_embed  # Flag to adapt the patch embedding layer
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None, adapt=False, HQ=False, IR=False, selfsup=False, increase_resolution=False, adapt_patch_embed=False):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        adapt=adapt,
        HQ=HQ,
        IR=IR,
        mlp_ratio_adapter=0.25,  # Default value for mlp_ratio in SAM
        dropout_prob=0.0,  # Default value for dropout in SAM
        selfsup=selfsup,  # Flag for self-supervised training
        increase_resolution=increase_resolution,  # Enable resolution increase for SAM
        adapt_patch_embed=adapt_patch_embed  # Flag to adapt the patch embedding layer
    )


def build_sam_vit_b(checkpoint=None, adapt=False, HQ=False, IR=False, mlp_ratio_adapter=0.25, dropout_prob=0.0, selfsup=False, increase_resolution=False, adapt_patch_embed=False):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        adapt=adapt,
        HQ=HQ,  
        IR=IR,
        mlp_ratio_adapter= mlp_ratio_adapter,  # Default value for mlp_ratio in SAM
        dropout_prob= dropout_prob,  # Default value for dropout in SAM  
        selfsup=selfsup,  # Flag for self-supervised training
        increase_resolution=increase_resolution,  # Enable resolution increase for SAM
        adapt_patch_embed=adapt_patch_embed  # Flag to adapt the patch embedding layer
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    adapt=False,
    HQ=False,
    IR=False,
    mlp_ratio_adapter=0.25,  # Default value for mlp_ratio in SAM
    dropout_prob=0.0,  # Default value for dropout in SAM
    selfsup = False,
    increase_resolution=False,
    adapt_patch_embed=False
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    decoder = MaskDecoderHQ if HQ else MaskDecoder
    encoder = ImageEncoderIrViT if IR else ImageEncoderViT
    sam = Sam(
        image_encoder=encoder(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            adapter =adapt,
            HQ=HQ,
            mlp_ratio_adapter= mlp_ratio_adapter,  
            dropout_prob= dropout_prob, 
            selfsup=selfsup,
            adapt_patch_embed=adapt_patch_embed
        ),

        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=decoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            increase_resolution=increase_resolution,  # Enable resolution increase for SAM
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        # Create a new state dictionary with only the parameters that exist in the model
        # Filter out keys with shape mismatches
        current_state = sam.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k in current_state and current_state[k].shape == v.shape
        }
        
        missing_keys, unexpected_keys = sam.load_state_dict(filtered_state_dict, strict=False)


        #missing_keys, unexpected_keys = sam.load_state_dict(state_dict, strict=False)


        ## Print missing and unexpected keys
        #print("Missing keys:", missing_keys)
        #print("Unexpected keys:", unexpected_keys)
        
    return sam
