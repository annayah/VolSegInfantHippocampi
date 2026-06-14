# Dependencies imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any


# Blocks/layers
def _make_scratch_3d(in_shape: List[int], out_shape: int, groups: int = 1):
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv3d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv3d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv3d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv3d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


# 3D Wrapper around frozen DINOv2 Encoder
class SegDINO3DEncoder(nn.Module):
    def __init__(
        self,
        dinov3_backbone: nn.Module,
        encoder_size: str = 'base',
        dino_input_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self.dinov3_backbone = dinov3_backbone
        self.encoder_size = encoder_size
        self.dino_input_size = dino_input_size
        self.patch_size = patch_size
        
        self.embed_dim = dinov3_backbone.embed_dim
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23],
        }
        
        # Learnable depth embedding for 3D positional encoding
        self.depth_embed = nn.Parameter(torch.zeros(1, 1, 64, self.embed_dim))
        nn.init.trunc_normal_(self.depth_embed, std=0.02)
        
        # Freeze the DINOv2 backbone
        self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.dinov3_backbone.parameters():
            param.requires_grad = False
        self.dinov3_backbone.eval()

    def _get_depth_embed(self, depth: int) -> torch.Tensor:
        depth_embed = F.interpolate(
            self.depth_embed.permute(0, 3, 1, 2),
            size=(1, depth),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)
        return depth_embed.squeeze(1)
    
    def get_intermediate_layers(
        self, x: torch.Tensor, 
        n: List[int] = None) -> List[torch.Tensor]:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        
        b, c, d, h, w = x.shape
        
        # Step 1: Unboxing - Reshape 3D volume into batch of 2D slices
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        
        # Resize slices to match DINOv2 expected input size if necessary
        original_h, original_w = h, w
        if h != self.dino_input_size or w != self.dino_input_size:
            x_2d = F.interpolate(
                x_2d, 
                size=(self.dino_input_size, self.dino_input_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Step 2: Extract features using frozen DINOv2 backbone
        with torch.no_grad():
            features_2d = self.dinov3_backbone.get_intermediate_layers(
                x_2d, 
                n=self.intermediate_layer_idx[self.encoder_size],
                reshape=False
            )
        
        # Step 3: Boxing - Reassemble the 2D features into 3D volumes
        features_3d = []
        num_patches_h = self.dino_input_size // self.patch_size
        num_patches_w = self.dino_input_size // self.patch_size
        
        depth_embed = self._get_depth_embed(d)
        
        for feat_2d in features_2d:
            _, n, c = feat_2d.shape

            feat_3d = feat_2d.reshape(b, d, num_patches_h, num_patches_w, c)
            
            feat_3d = feat_3d + depth_embed.unsqueeze(2).unsqueeze(3)
            
            feat_3d = feat_3d.reshape(b, d * num_patches_h * num_patches_w, c)
            
            features_3d.append(feat_3d)
        
        return features_3d


# 3D DPT Decoder Head
class DPTHead3D(nn.Module):
    def __init__(
        self, 
        nclass: int,
        in_channels: int, 
        features: int = 256, 
        out_channels: List[int] = [256, 512, 1024, 1024],
    ):
        super().__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.scratch = _make_scratch_3d(out_channels, features)

        self.proj = nn.ConvTranspose3d(
            features, features,
            kernel_size=4, stride=4, padding=0, bias=False,
        )

        self.fusion_refinement = nn.Sequential(
            nn.Conv3d(features * 4, features * 4, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(features * 4, features * 4, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features * 4),
            nn.ReLU(inplace=True),
        )

        self.output_conv = nn.Conv3d(
            features * 4, nclass,
            kernel_size=1, stride=1, padding=0,
        )
    
    def forward(
        self, 
        out_features: List[torch.Tensor], 
        patch_d: int, 
        patch_h: int, 
        patch_w: int,
    ) -> torch.Tensor:

        out = []
        for i, x in enumerate(out_features):
            b, _, c = x.shape
            x = x.permute(0, 2, 1).reshape(b, c, patch_d, patch_h, patch_w)
            x = self.projects[i](x)
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        layer_1_rn = self.proj(layer_1_rn)

        target_dhw = layer_1_rn.shape[2:]

        layer_2_up = F.interpolate(layer_2_rn, size=target_dhw, mode="trilinear", align_corners=True)
        layer_3_up = F.interpolate(layer_3_rn, size=target_dhw, mode="trilinear", align_corners=True)
        layer_4_up = F.interpolate(layer_4_rn, size=target_dhw, mode="trilinear", align_corners=True)

        fused = torch.cat([layer_1_rn, layer_2_up, layer_3_up, layer_4_up], dim=1)

        refined_fusion = self.fusion_refinement(fused)

        out = self.output_conv(refined_fusion)
        return out


# End-to-end 3D SegDINO
class SegDINO3D(nn.Module):
    def __init__(
        self, 
        dinov3_backbone: nn.Module,
        encoder_size: str = 'base', 
        nclass: int = 1,
        features: int = 128, 
        out_channels: List[int] = [96, 192, 384, 768], 
        use_bn: bool = False,
        dino_input_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        
        self.encoder_size = encoder_size
        self.nclass = nclass
        self.dino_input_size = dino_input_size
        self.patch_size = patch_size
        
        # Create the 3D encoder wrapper
        self.encoder = SegDINO3DEncoder(
            dinov3_backbone=dinov3_backbone,
            encoder_size=encoder_size,
            dino_input_size=dino_input_size,
            patch_size=patch_size,
        )
        
        # Create the 3D DPT head
        self.head = DPTHead3D(
            nclass=nclass,
            in_channels=self.encoder.embed_dim,
            features=features,
            out_channels=out_channels,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, c, d, h, w = x.shape
        
        patch_d = d
        patch_h = self.dino_input_size // self.patch_size
        patch_w = self.dino_input_size // self.patch_size

        features = self.encoder(x)
        
        out = self.head(features, patch_d, patch_h, patch_w)
        
        out = F.interpolate(out, size=(d, h, w), mode='trilinear', align_corners=True)
        
        return out
    

# Factory function
def create_segdino3d(
    dino_repo_path: str,
    dino_weights_path: str,
    encoder_size: str = 'base',
    nclass: int = 1,
    features: int = 128,
    dino_input_size: int = 224,
) -> SegDINO3D:

    if encoder_size == 'small':
        dinov3_backbone = torch.hub.load(
            dino_repo_path, 
            'dinov3_vits16', 
            source='local', 
            weights=dino_weights_path
        )
        out_channels = [96, 192, 384, 384]
    elif encoder_size == 'base':
        dinov3_backbone = torch.hub.load(
            dino_repo_path, 
            'dinov3_vitb16', 
            source='local', 
            weights=dino_weights_path
        )
        out_channels = [96, 192, 384, 768]
    elif encoder_size == 'large':
        dinov3_backbone = torch.hub.load(
            dino_repo_path, 
            'dinov3_vitl16', 
            source='local', 
            weights=dino_weights_path
        )
        out_channels = [256, 512, 1024, 1024]
    else:
        raise ValueError(f"Unknown encoder size: {encoder_size}")
    
    model = SegDINO3D(
        dinov3_backbone=dinov3_backbone,
        encoder_size=encoder_size,
        nclass=nclass,
        features=features,
        out_channels=out_channels,
        dino_input_size=dino_input_size,
    )
    
    return model