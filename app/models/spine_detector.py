import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2

class MCBAM(nn.Module):
    """多上下文注意力模块（添加空间金字塔）"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 空间金字塔卷积
        self.pyramid_conv = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=1, dilation=1),
            nn.Conv2d(channels, channels//4, 3, padding=2, dilation=2),
            nn.Conv2d(channels, channels//4, 3, padding=4, dilation=4),
        )
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        # 空间注意力
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        # 通道注意力
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = torch.sigmoid(self.conv(spatial_att))

        return x * channel_att * spatial_att


class SpineKeypointDetector(nn.Module):
    """脊柱关键点检测模型（改进UNet架构）"""

    def __init__(self, num_keypoints=17):
        super().__init__()
        # 使用MCBAM模块
        self.mcba1 = MCBAM(64)
        self.mcba2 = MCBAM(128)
        self.mcba3 = MCBAM(256)
        
        # UNet编码器
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self.mcba1
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.mcba2
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self.mcba3
        )

        # UNet解码器
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.mcba2
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self.mcba1
        )

        # 多任务输出头
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints * 2, 1)  # (x,y)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints, 1)  # 置信度
        )
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)  # 分割掩膜
        )
        
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, 1)  # 热力图
        )
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码过程
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        # 解码过程
        dec3 = self.up3(enc3)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.decoder2(dec2)

        # 多任务预测
        keypoints_coord = self.keypoint_head(dec2)
        keypoints_conf = self.confidence_head(dec2)
        seg_mask = self.seg_head(enc3)
        heatmaps = self.heatmap_head(dec3)
        
        # 重塑坐标输出
        batch_size = x.shape[0]
        num_keypoints = keypoints_coord.shape[1] // 2
        
        # 将坐标输出从 [B, K*2, 1, 1] 调整为 [B, K, 2]
        keypoints_coord = keypoints_coord.view(batch_size, num_keypoints, 2)
        
        # 将置信度输出从 [B, K, 1, 1] 调整为 [B, K, 1]
        keypoints_conf = keypoints_conf.view(batch_size, num_keypoints, 1)
        
        # 将坐标归一化到 [0,1] 范围
        keypoints_coord = torch.sigmoid(keypoints_coord)
        
        # 多任务输出
        return {
            'coordinates': keypoints_coord,
            'confidence': keypoints_conf,
            'segmentation': seg_mask,
            'heatmaps': heatmaps
        }
        
    @staticmethod
    def preprocess_image(image, target_size=(512, 512)):
        """预处理图像用于模型输入
        
        Args:
            image: 输入图像 (numpy数组)
            target_size: 目标尺寸
            
        Returns:
            预处理后的图像张量
        """
        # 调整大小
        image = cv2.resize(image, target_size)
        
        # 转换为RGB
        if len(image.shape) == 2:  # 灰度图像
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA图像
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:  # BGR图像
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # 归一化
        image = image.astype(np.float32) / 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = (image - mean) / std
        
        # 转换为PyTorch张量
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image
        
    @staticmethod
    def detect_spine_keypoints(model, image):
        """检测脊柱关键点
        
        Args:
            model: 脊柱关键点检测模型
            image: 输入图像
            
        Returns:
            关键点坐标和置信度
        """
        # 预处理图像
        input_tensor = SpineKeypointDetector.preprocess_image(image)
        
        # 推理
        with torch.no_grad():
            outputs = model(input_tensor)
            
        # 获取关键点坐标和置信度
        keypoints_coord = outputs['coordinates'][0].cpu().numpy()  # [K, 2]
        keypoints_conf = outputs['confidence'][0].cpu().numpy()    # [K, 1]
        
        # 将关键点坐标转换回原始图像尺寸
        h, w = image.shape[:2]
        keypoints_scaled = keypoints_coord.copy()
        keypoints_scaled[:, 0] *= w
        keypoints_scaled[:, 1] *= h
        
        # 合并坐标和置信度
        keypoints = np.concatenate([keypoints_scaled, keypoints_conf], axis=1)
        
        return keypoints