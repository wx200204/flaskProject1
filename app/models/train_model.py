import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from spine_detector import SpineKeypointDetector


class SpineDataset(Dataset):
    """脊柱关键点数据集"""
    
    def __init__(self, data_dir, annotation_file, transform=None):
        """
        初始化数据集
        
        Args:
            data_dir: 图像目录
            annotation_file: 标注文件路径
            transform: 数据增强转换
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 加载标注
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        # 图像文件列表
        self.image_files = list(self.annotations.keys())
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        # 获取图像文件名
        img_file = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_file)
        
        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取关键点标注
        keypoints = self.annotations[img_file]['keypoints']
        keypoints = np.array(keypoints, dtype=np.float32)
        
        # 归一化关键点坐标
        h, w = image.shape[:2]
        keypoints[:, 0] /= w
        keypoints[:, 1] /= h
        
        # 应用数据增强
        if self.transform:
            image, keypoints = self.transform(image, keypoints)
            
        # 转换为张量
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        
        return torch.from_numpy(image), torch.from_numpy(keypoints)


class KeypointLoss(nn.Module):
    """关键点检测损失函数"""
    
    def __init__(self, use_visibility=True):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.use_visibility = use_visibility
        
    def forward(self, pred, target):
        """
        计算损失
        
        Args:
            pred: 预测关键点 [batch_size, num_keypoints, 3]
            target: 目标关键点 [batch_size, num_keypoints, 3]
            
        Returns:
            损失值
        """
        # 坐标损失
        coord_loss = self.mse(pred[:, :, :2], target[:, :, :2])
        
        if self.use_visibility and pred.shape[2] > 2 and target.shape[2] > 2:
            # 使用可见性权重
            visibility = target[:, :, 2:3]
            coord_loss = coord_loss * visibility
            
            # 可见性损失
            vis_loss = self.mse(pred[:, :, 2:3], target[:, :, 2:3])
            
            # 总损失
            loss = coord_loss.mean() + 0.1 * vis_loss.mean()
        else:
            loss = coord_loss.mean()
            
        return loss


def train_model(args):
    """训练脊柱关键点检测模型"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据集
    train_dataset = SpineDataset(args.data_dir, args.train_annotations)
    val_dataset = SpineDataset(args.data_dir, args.val_annotations)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 创建模型
    model = SpineKeypointDetector(num_keypoints=args.num_keypoints)
    
    # 如果指定了预训练模型，加载权重
    if args.pretrained:
        state_dict = torch.load(args.pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    
    # 移动模型到设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = KeypointLoss(use_visibility=True)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]'):
            # 移动数据到设备
            images = images.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
            train_batches += 1
        
        # 计算平均训练损失
        train_loss /= train_batches
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]'):
                # 移动数据到设备
                images = images.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # 累计损失
                val_loss += loss.item()
                val_batches += 1
        
        # 计算平均验证损失
        val_loss /= val_batches
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印进度
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
        # 保存最新模型
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss
        }, os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))
    
    # 导出为TorchScript模型
    model.eval()
    example = torch.rand(1, 3, 512, 512).to(device)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(os.path.join(args.output_dir, 'model_scripted.pth'))
    
    print(f'Training completed. Models saved to {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train spine keypoint detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--train_annotations', type=str, required=True, help='Path to training annotations JSON')
    parser.add_argument('--val_annotations', type=str, required=True, help='Path to validation annotations JSON')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for models')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_keypoints', type=int, default=17, help='Number of keypoints')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained model')
    
    args = parser.parse_args()
    train_model(args) 