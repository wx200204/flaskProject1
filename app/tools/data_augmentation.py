import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
import random
import shutil
from tqdm import tqdm


class SpineDataAugmenter:
    """脊柱X光片数据增强工具"""
    
    def __init__(self, image_dir, annotation_file, output_dir, output_annotation_file, debug_dir='debug_aug'):
        # 创建调试目录
        self.debug_dir = Path(debug_dir)
        os.makedirs(self.debug_dir, exist_ok=True)
        """
        初始化数据增强工具
        
        Args:
            image_dir: 原始图像目录
            annotation_file: 原始标注文件路径
            output_dir: 输出图像目录
            output_annotation_file: 输出标注文件路径
        """
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.output_dir = Path(output_dir)
        self.output_annotation_file = Path(output_annotation_file)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载标注
        with open(self.annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        # 初始化输出标注
        self.output_annotations = {}
        
    def apply_augmentations(self, num_augmentations_per_image=5):
        """
        对每张图像应用多种数据增强
        
        Args:
            num_augmentations_per_image: 每张图像生成的增强版本数量
        """
        # 新增医学专用增强方法
        self.medical_augmentations = [
            self.adjust_vertebra_spacing,
            self.simulate_xray_scatter,
            self.add_limb_interference  # 新增肢体干扰增强
        ]
        # 首先复制原始图像和标注
        for img_file, annotation in tqdm(self.annotations.items(), desc="复制原始数据"):
            # 复制图像
            src_path = self.image_dir / img_file
            dst_path = self.output_dir / img_file
            shutil.copy2(src_path, dst_path)
            
            # 复制标注
            self.output_annotations[img_file] = annotation
        
        # 对每张图像应用增强
        for img_file, annotation in tqdm(self.annotations.items(), desc="应用数据增强"):
            # 读取图像
            img_path = self.image_dir / img_file
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue
                
            # 获取关键点
            keypoints = np.array(annotation['keypoints'])
            
            # 应用多种增强
            for i in range(num_augmentations_per_image):
                # 生成增强图像文件名
                base_name, ext = os.path.splitext(img_file)
                aug_img_file = f"{base_name}_aug_{i+1}{ext}"
                
                # 随机选择增强方法
                # 50%概率使用医学专用增强
                if random.random() < 0.5 and self.medical_augmentations:
                    aug_method = random.choice(self.medical_augmentations)
                else:
                    aug_method = random.choice([
                        self.random_rotation,
                        self.random_brightness_contrast,
                        self.random_noise,
                        self.random_flip,
                        self.random_scale
                    ])
                
                # 应用增强
                aug_image, aug_keypoints = aug_method(image.copy(), keypoints.copy())
                
                # 保存增强图像
                aug_img_path = self.output_dir / aug_img_file
                cv2.imwrite(str(aug_img_path), aug_image)
                
                # 保存增强标注
                self.output_annotations[aug_img_file] = {
                    'keypoints': aug_keypoints.tolist(),
                    'image_width': aug_image.shape[1],
                    'image_height': aug_image.shape[0]
                }
        
        # 保存输出标注
        with open(self.output_annotation_file, 'w') as f:
            json.dump(self.output_annotations, f, indent=2)
            
        print(f"数据增强完成。原始图像: {len(self.annotations)}, 增强后图像: {len(self.output_annotations)}")
    
    def random_rotation(self, image, keypoints, max_angle=15):
        """
        随机旋转图像和关键点
        
        Args:
            image: 输入图像
            keypoints: 关键点坐标 [N, 3] (x, y, visibility)
            max_angle: 最大旋转角度
            
        Returns:
            旋转后的图像和关键点
        """
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        # 随机角度
        angle = random.uniform(-max_angle, max_angle)
        
        # 旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 旋转图像
        rotated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # 旋转关键点
        rotated_keypoints = keypoints.copy()
        for i in range(len(keypoints)):
            if keypoints[i, 2] > 0:  # 只旋转可见的关键点
                x, y = keypoints[i, 0], keypoints[i, 1]
                
                # 应用旋转矩阵
                new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
                new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
                
                rotated_keypoints[i, 0] = new_x
                rotated_keypoints[i, 1] = new_y
        
        return rotated_image, rotated_keypoints
    
    def random_brightness_contrast(self, image, keypoints, alpha_range=(0.8, 1.2), beta_range=(-30, 30)):
        """
        随机调整亮度和对比度
        
        Args:
            image: 输入图像
            keypoints: 关键点坐标
            alpha_range: 对比度调整范围
            beta_range: 亮度调整范围
            
        Returns:
            调整后的图像和原始关键点
        """
        # 随机对比度和亮度参数
        alpha = random.uniform(*alpha_range)
        beta = random.uniform(*beta_range)
        
        # 应用变换: new_pixel = alpha * old_pixel + beta
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # 关键点不变
        return adjusted_image, keypoints
    
    class ClothingNoiseGenerator:
    """医疗影像衣物噪声模拟器"""
    def __init__(self):
        self.pattern_types = ['stripes', 'dots', 'mesh']
        self.noise_intensity = (0.1, 0.3)

    def add_clothing_noise(self, image):
        """添加模拟衣物纹理的结构化噪声"""
        h, w = image.shape[:2]
        
        # 随机选择噪声模式
        pattern = np.random.choice(self.pattern_types)
        noise_layer = np.zeros((h, w), dtype=np.uint8)
        
        if pattern == 'stripes':
            # 生成随机条纹
            for _ in range(np.random.randint(3, 8)):
                thickness = np.random.randint(2, 5)
                cv2.line(noise_layer, 
                        (np.random.randint(0, w), np.random.randint(0, h)),
                        (np.random.randint(0, w), np.random.randint(0, h)),
                        255, thickness)
        elif pattern == 'dots':
            # 生成随机点阵
            dots = np.random.rand(h, w) > 0.95
            noise_layer[dots] = 255
        else: # mesh
            # 生成网格状噪声
            grid_size = np.random.randint(15, 30)
            noise_layer[::grid_size, :] = 255
            noise_layer[:, ::grid_size] = 255
        
        # 应用噪声到原图
        alpha = np.random.uniform(*self.noise_intensity)
        return cv2.addWeighted(image, 1-alpha, 
                             cv2.cvtColor(noise_layer, cv2.COLOR_GRAY2BGR), 
                             alpha, 0)

def random_noise(self, image, keypoints, noise_level=15):
        """
        添加随机噪声
        
        Args:
            image: 输入图像
            keypoints: 关键点坐标
            noise_level: 噪声级别
            
        Returns:
            添加噪声后的图像和原始关键点
        """
        # 生成高斯噪声
        noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
        
        # 添加噪声
        noisy_image = cv2.add(image, noise.astype(np.int16))
        
        # 裁剪到有效范围
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        # 关键点不变
        return noisy_image, keypoints
    
    def adjust_vertebra_spacing(self, image, keypoints, max_shift=0.15):
        """椎骨间距调整增强
        
        Args:
            image: 输入图像
            keypoints: 关键点坐标 [N,3]
            max_shift: 最大位移比例
        """
        # 仅对可见关键点操作
        visible_points = keypoints[keypoints[:,2] > 0]
        if len(visible_points) < 2:
            return image, keypoints

        # 计算相邻关键点间距
        diffs = np.diff(visible_points[:, :2], axis=0)
        avg_spacing = np.mean(np.linalg.norm(diffs, axis=1))
        
        # 生成随机位移
        shift = np.random.uniform(-max_shift, max_shift, size=(len(visible_points), 2)) * avg_spacing
        
        # 应用位移（中间关键点位移较大）
        for i in range(1, len(visible_points)-1):
            visible_points[i, :2] += shift[i] * (i/len(visible_points))
        
        # 更新关键点
        keypoints[keypoints[:,2] > 0] = visible_points
        
        # 可视化调试
        if random.random() < 0.1:  # 10%概率保存调试图像
            debug_img = image.copy()
            for pt in visible_points:
                cv2.circle(debug_img, tuple(pt[:2].astype(int)), 3, (0,255,0), -1)
            cv2.imwrite(str(self.debug_dir/'vertebra_shift.jpg'), debug_img)
        
        return image, keypoints

    def simulate_xray_scatter(self, image, keypoints, intensity_range=(0.1, 0.3)):
        """X光散射效果模拟
        
        Args:
            image: 输入图像
            keypoints: 关键点坐标
            intensity_range: 散射强度范围
        """
        # 生成散射噪声
        h, w = image.shape[:2]
        scatter = np.zeros((h, w), dtype=np.float32)
        
        # 随机散射中心（靠近关键点）
        for _ in range(np.random.randint(3, 6)):
            cx = np.random.choice(keypoints[keypoints[:,2]>0][:,0].astype(int))
            cy = np.random.choice(keypoints[keypoints[:,2]>0][:,1].astype(int))
            kernel_size = np.random.randint(50, 150)
            scatter += cv2.getGaussianKernel(kernel_size, 0) @ cv2.getGaussianKernel(kernel_size, 0).T
        
        # 归一化并应用强度
        scatter = cv2.normalize(scatter, None, 0, 1, cv2.NORM_MINMAX)
        intensity = np.random.uniform(*intensity_range)
        scatter = (scatter * intensity * 255).astype(np.uint8)
        
        # 应用散射效果
        blended = cv2.addWeighted(image, 1 - intensity, cv2.cvtColor(scatter, cv2.COLOR_GRAY2BGR), intensity, 0)
        
        # 可视化调试
        if random.random() < 0.1:
            cv2.imwrite(str(self.debug_dir/'xray_scatter.jpg'), blended)
        
        return blended, keypoints

    def add_limb_interference(self, image, keypoints):
        """添加模拟肢体结构干扰"""
        h, w = image.shape[:2]
        
        # 生成随机肢体形状噪声
        noise = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(random.randint(1,3)):
            start = (random.randint(0,w), random.randint(0,h))
            end = (random.randint(0,w), random.randint(0,h))
            cv2.line(noise, start, end, (random.randint(150,250),)*3, 
                    thickness=random.randint(10,30))
            
        # 使用泊松融合混合噪声
        blended = cv2.seamlessClone(noise, image, 
                                   np.full((h,w),255,np.uint8),
                                   (w//2,h//2), cv2.NORMAL_CLONE)
        
        # 10%概率保存调试图像
        if random.random() < 0.1:
            cv2.imwrite(str(self.debug_dir/'limb_interference.jpg'), blended)
        
        return blended, keypoints

    def random_flip(self, image, keypoints):
        """
        水平翻转图像和关键点
        
        Args:
            image: 输入图像
            keypoints: 关键点坐标
            
        Returns:
            翻转后的图像和关键点
        """
        h, w = image.shape[:2]
        
        # 水平翻转图像
        flipped_image = cv2.flip(image, 1)  # 1表示水平翻转
        
        # 翻转关键点
        flipped_keypoints = keypoints.copy()
        for i in range(len(keypoints)):
            if keypoints[i, 2] > 0:  # 只翻转可见的关键点
                flipped_keypoints[i, 0] = w - keypoints[i, 0]
        
        return flipped_image, flipped_keypoints
    
    def random_scale(self, image, keypoints, scale_range=(0.9, 1.1)):
        """
        随机缩放图像和关键点
        
        Args:
            image: 输入图像
            keypoints: 关键点坐标
            scale_range: 缩放范围
            
        Returns:
            缩放后的图像和关键点
        """
        h, w = image.shape[:2]
        
        # 随机缩放因子
        scale = random.uniform(*scale_range)
        
        # 计算新尺寸
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 缩放图像
        scaled_image = cv2.resize(image, (new_w, new_h))
        
        # 如果缩放后图像尺寸小于原始尺寸，需要填充
        if new_h < h or new_w < w:
            # 创建空白图像
            padded_image = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 计算填充位置
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            
            # 填充图像
            padded_image[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled_image
            
            # 调整关键点坐标
            scaled_keypoints = keypoints.copy()
            for i in range(len(keypoints)):
                if keypoints[i, 2] > 0:  # 只调整可见的关键点
                    scaled_keypoints[i, 0] = keypoints[i, 0] * scale + pad_w
                    scaled_keypoints[i, 1] = keypoints[i, 1] * scale + pad_h
            
            return padded_image, scaled_keypoints
        
        # 如果缩放后图像尺寸大于原始尺寸，需要裁剪
        elif new_h > h or new_w > w:
            # 计算裁剪位置
            crop_h = (new_h - h) // 2
            crop_w = (new_w - w) // 2
            
            # 裁剪图像
            cropped_image = scaled_image[crop_h:crop_h+h, crop_w:crop_w+w]
            
            # 调整关键点坐标
            scaled_keypoints = keypoints.copy()
            for i in range(len(keypoints)):
                if keypoints[i, 2] > 0:  # 只调整可见的关键点
                    scaled_keypoints[i, 0] = keypoints[i, 0] * scale - crop_w
                    scaled_keypoints[i, 1] = keypoints[i, 1] * scale - crop_h
            
            return cropped_image, scaled_keypoints
        
        # 如果缩放后图像尺寸与原始尺寸相同，直接返回
        else:
            # 调整关键点坐标
            scaled_keypoints = keypoints.copy()
            for i in range(len(keypoints)):
                if keypoints[i, 2] > 0:  # 只调整可见的关键点
                    scaled_keypoints[i, 0] = keypoints[i, 0] * scale
                    scaled_keypoints[i, 1] = keypoints[i, 1] * scale
            
            return scaled_image, scaled_keypoints


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='脊柱X光片数据增强工具')
    parser.add_argument('--image_dir', type=str, required=True, help='原始图像目录')
    parser.add_argument('--annotation_file', type=str, required=True, help='原始标注文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出图像目录')
    parser.add_argument('--output_annotation', type=str, required=True, help='输出标注文件路径')
    parser.add_argument('--num_augmentations', type=int, default=5, help='每张图像生成的增强版本数量')
    
    args = parser.parse_args()
    
    # 创建并运行数据增强工具
    augmenter = SpineDataAugmenter(
        args.image_dir, 
        args.annotation_file, 
        args.output_dir, 
        args.output_annotation
    )
    augmenter.apply_augmentations(args.num_augmentations)


if __name__ == '__main__':
    main()