import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import roi_align
from torchvision.models import resnet50


class MedicalImageProcessor:
    """医学图像专用预处理模块"""

    def __init__(self):
        self.norm_params = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        医学图像标准化处理流程
        Args:
            image: 输入BGR图像 (H, W, 3)
        Returns:
            标准化后的Tensor (1, 3, 512, 512)
        """
        # 颜色空间转换
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 动态对比度增强
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        limg = clahe.apply(l)
        enhanced = cv2.merge((limg, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        # 标准化处理
        tensor = torch.from_numpy(enhanced_rgb).permute(2, 0, 1).float()
        tensor = nn.functional.interpolate(tensor.unsqueeze(0), size=512)
        tensor = (tensor / 255.0 - self.norm_params['mean']) / self.norm_params['std']
        return tensor


class SpineKeypointDetector(nn.Module):
    """改进型脊柱关键点检测网络"""

    def __init__(self, num_keypoints=24):
        super().__init__()
        base = resnet50(weights='IMAGENET1K_V2')
        self.feature_extractor = nn.Sequential(*list(base.children())[:-2])

        # 特征金字塔增强
        self.fpn = nn.ModuleList([
            nn.Conv2d(2048, 256, 1),
            nn.Conv2d(1024, 256, 1),
            nn.Conv2d(512, 256, 1)
        ])

        # 关键点回归头
        self.reg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints * 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming初始化关键层"""
        for m in self.reg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征提取
        feats = self.feature_extractor(x)

        # FPN多尺度融合
        pyramid_feats = []
        for i, f in enumerate(self.fpn):
            if i == 0:
                feat = f(feats)
            else:
                feat = f(feats[-i])
            pyramid_feats.append(nn.functional.interpolate(feat, scale_factor=2 ** i))

        fused_feat = torch.sum(torch.stack(pyramid_feats), dim=0)

        # 关键点预测
        return self.reg_head(fused_feat)