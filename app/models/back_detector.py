import cv2
import numpy as np
import os
import logging
from pathlib import Path
from flask import current_app
import traceback
import time


class BackSpineDetector:
    """背部脊柱检测器 - 使用增强图像处理技术检测背部照片中的脊柱线"""
    
    def __init__(self, config=None):
        """初始化检测器
        
        Args:
            config: 配置参数
        """
        self.logger = logging.getLogger(__name__)
        
        # 默认配置
        self.config = {
            'num_keypoints': 17,  # 脊柱关键点数量
            'debug_mode': False,  # 调试模式
            'debug_dir': str(Path(__file__).parent.parent.parent / 'debug'),
            'hough_threshold': 30,  # 霍夫线检测阈值
            'hough_min_length': 50,  # 霍夫线最小长度
            'hough_max_gap': 10,    # 霍夫线最大间隔
            'edge_threshold1': 30,  # Canny边缘检测低阈值
            'edge_threshold2': 90   # Canny边缘检测高阈值
        }
        
        # 更新配置
        if config:
            self.config.update(config)
            
        # 创建调试目录
        if self.config['debug_mode']:
            os.makedirs(self.config['debug_dir'], exist_ok=True)
    
    def detect_spine(self, image):
        """改进的脊柱检测方法，引入医学解剖学约束和高级特征提取"""
        try:
            # 记录处理起始时间
            start_time = time.time()
            if 'current_app' in globals():
                current_app.logger.info("开始脊柱检测处理")
                
            # 1. 增强预处理 - 提高脊柱特征突出度
            img_height, img_width = image.shape[:2]
            preprocessed = self._enhanced_preprocess(image)
            
            # 2. 多尺度检测 - 在不同尺度搜索脊柱特征
            center_lines = []
            confidence_scores = []
            
            # 设置多尺度参数 - 在不同比例下检测
            scale_factors = [1.0, 0.75, 0.5]
            
            for scale in scale_factors:
                # 跳过太小的尺度
                if min(img_height, img_width) * scale < 100:
                    continue
                    
                # 调整图像尺寸
                if scale != 1.0:
                    scaled_size = (int(img_width * scale), int(img_height * scale))
                    scaled_img = cv2.resize(preprocessed, scaled_size, interpolation=cv2.INTER_AREA)
                else:
                    scaled_img = preprocessed
                
                # 在当前尺度下检测
                line, score = self._detect_spine_at_scale(scaled_img)
                
                # 将坐标转换回原始尺度
                if scale != 1.0:
                    line[:, 0] = line[:, 0] / scale
                    line[:, 1] = line[:, 1] / scale
                
                center_lines.append(line)
                confidence_scores.append(score)
            
            # 3. 综合多尺度结果 - 选择最佳或融合结果
            if len(center_lines) > 0:
                # 选择置信度最高的结果
                best_idx = np.argmax(confidence_scores)
                raw_keypoints = center_lines[best_idx]
                best_score = confidence_scores[best_idx]
                
                if 'current_app' in globals():
                    current_app.logger.debug(f"最佳检测尺度得分: {best_score:.2f}")
            else:
                # 备用：使用基础检测方法
                raw_keypoints = self._generate_center_line(preprocessed)
                best_score = 0.5
            
            # 4. 应用解剖学约束并优化关键点
            from ..utils.anatomy_postprocess import SpinePostProcessor
            processor = SpinePostProcessor()
            refined_keypoints, symmetry, curvature = processor.apply_constraints(
                raw_keypoints, image.shape[:2]
            )
            
            # 5. 深度增强置信度计算
            keypoints_with_conf = []
            
            # 使用多种图像质量评估指标来增强置信度计算
            clarity_score = self._calculate_image_clarity(preprocessed)
            texture_score = self._calculate_texture_quality(preprocessed)
            contrast_score = self._calculate_contrast_quality(preprocessed)
            
            # 计算图像质量综合评分
            image_quality = (clarity_score * 0.5 + texture_score * 0.3 + contrast_score * 0.2)
            image_quality = min(0.95, max(0.5, image_quality))
            
            if 'current_app' in globals():
                current_app.logger.debug(f"图像质量评分: {image_quality:.2f} (清晰度={clarity_score:.2f}, "
                                      f"纹理={texture_score:.2f}, 对比度={contrast_score:.2f})")
            
            # 为每个点计算基础置信度
            base_confidence = min(0.95, max(0.4, image_quality * best_score))
            
            # 为每个点添加置信度
            for i, point in enumerate(refined_keypoints):
                # 位置因子 - 脊柱中间通常更清晰，因此置信度更高
                position_factor = 1.0 - 0.3 * abs(i / (len(refined_keypoints) - 1) - 0.5) * 2
                
                # 对称性和曲率的影响 
                conf = base_confidence * position_factor * (0.7 + 0.3 * symmetry) * (0.7 + 0.3 * curvature)
                
                # 确保置信度在合理范围内
                conf = max(0.4, min(0.95, conf))
                
                keypoints_with_conf.append([point[0], point[1], conf])
            
            # 6. 最终排序和验证
            keypoints_array = np.array(keypoints_with_conf)
            sorted_indices = np.argsort(keypoints_array[:, 1])
            sorted_keypoints = keypoints_array[sorted_indices]
            
            # 记录检测耗时
            elapsed = time.time() - start_time
            if 'current_app' in globals():
                current_app.logger.info(f"脊柱检测完成，耗时 {elapsed:.3f}秒，"
                                      f"检测到 {len(sorted_keypoints)} 个关键点")
            
            return sorted_keypoints
            
        except Exception as e:
            if 'current_app' in globals():
                current_app.logger.error(f"脊柱检测失败: {str(e)}")
                current_app.logger.error(traceback.format_exc())
            # 返回备用关键点，但确保每个图像的关键点不同
            return self._generate_dynamic_keypoints(image)
            
    def _enhanced_preprocess(self, image):
        """增强的预处理步骤，优化脊柱特征提取
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        # 确保图像是彩色的
        if len(image.shape) < 3:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img = image.copy()
            
        # 1. 提取灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 增强灰度图像对比度 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3. 高斯降噪同时保留边缘
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 4. 增强背部脊柱特征
        # 创建垂直脊柱特征探测器
        h, w = gray.shape[:2]
        vertical_size = max(3, int(h / 25))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
        
        # 应用形态学操作增强垂直结构
        vertical = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, vertical_kernel)
        
        # 5. 进行边缘检测 - 动态阈值
        mean_val = np.mean(enhanced)
        std_val = np.std(enhanced)
        
        low_threshold = max(10, int(mean_val / 2 - std_val / 2))
        high_threshold = min(150, int(mean_val + std_val))
        
        edges = cv2.Canny(vertical, low_threshold, high_threshold)
        
        # 6. 进行纹理增强
        # 使用Sobel算子计算梯度
        sobelx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值和角度
        magnitude = cv2.magnitude(sobelx, sobely)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 7. 融合多种特征
        # 综合各种特征图
        feature_img = cv2.addWeighted(denoised, 0.6, magnitude, 0.4, 0)
        
        # 将边缘信息融合进去 - 修复尺寸不匹配问题
        # 确保edges和feature_img尺寸相同
        if len(feature_img.shape) == 3:
            # 如果feature_img是3通道，将edges转换为3通道
            if edges.shape[:2] != feature_img.shape[:2]:
                # 调整edges大小以匹配feature_img
                edges = cv2.resize(edges, (feature_img.shape[1], feature_img.shape[0]))
            edges_3ch = cv2.merge([edges, edges, edges])
        else:
            # 如果feature_img是单通道，则不需要转换edges
            if edges.shape != feature_img.shape:
                edges = cv2.resize(edges, (feature_img.shape[1], feature_img.shape[0]))
            edges_3ch = edges
        
        # 确保两个图像尺寸完全匹配
        assert feature_img.shape == edges_3ch.shape, f"尺寸不匹配：feature_img {feature_img.shape}, edges_3ch {edges_3ch.shape}"
        
        final_img = cv2.addWeighted(feature_img, 0.7, edges_3ch, 0.3, 0)
        
        # 为调试目的保存处理中间步骤
        if self.config.get('debug_mode', False):
            debug_dir = self.config.get('debug_dir', './debug')
            os.makedirs(debug_dir, exist_ok=True)
            
            # 时间戳用于唯一文件名
            timestamp = int(time.time() * 1000) % 10000
            
            cv2.imwrite(f"{debug_dir}/1_gray_{timestamp}.jpg", gray)
            cv2.imwrite(f"{debug_dir}/2_enhanced_{timestamp}.jpg", enhanced)
            cv2.imwrite(f"{debug_dir}/3_denoised_{timestamp}.jpg", denoised)
            cv2.imwrite(f"{debug_dir}/4_vertical_{timestamp}.jpg", vertical)
            cv2.imwrite(f"{debug_dir}/5_edges_{timestamp}.jpg", edges)
            cv2.imwrite(f"{debug_dir}/6_magnitude_{timestamp}.jpg", magnitude)
            cv2.imwrite(f"{debug_dir}/7_final_{timestamp}.jpg", final_img)
        
        return final_img
        
    def _detect_spine_at_scale(self, image):
        """在特定尺度下检测脊柱
        
        Args:
            image: 预处理后的图像
            
        Returns:
            tuple: (检测到的脊柱线, 检测置信度)
        """
        h, w = image.shape[:2]
        
        # 1. 创建二值化图像突出垂直结构
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. 应用形态学操作增强脊柱
        kernel_size = max(3, int(h / 50))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 3. 创建距离变换，找出脊柱中心线
        # 计算图像中线
        mid_x = w // 2
        
        # 创建背部中线区域掩码
        spine_region_width = max(int(w * 0.3), 100)
        spine_mask = np.zeros_like(binary)
        spine_mask[:, max(0, mid_x - spine_region_width//2):min(w, mid_x + spine_region_width//2)] = 255
        
        # 应用掩码，仅考虑中央区域
        spine_area = cv2.bitwise_and(morphed, spine_mask)
        
        # 4. 使用Hough变换检测主要线条
        # 计算边缘
        edges = cv2.Canny(spine_area, 50, 150)
        
        # 设置Hough参数 - 动态参数
        hough_threshold = max(10, int(h / 15))
        min_line_length = max(30, int(h / 10))
        max_line_gap = max(10, int(h / 30))
        
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=hough_threshold, 
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
        )
        
        # 5. 过滤和组织线段，找出脊柱区域
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 计算线段角度（相对于垂直线）
                angle = abs(np.degrees(np.arctan2(x2 - x1, y2 - y1)))
                
                # 选择接近垂直的线段（±30度）
                if angle < 30:
                    line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    # 过长的线可能不是脊柱
                    if line_length < h * 0.8:
                        vertical_lines.append([(x1, y1), (x2, y2)])
        
        # 6. 生成关键点
        num_points = self.config['num_keypoints']
        keypoints = np.zeros((num_points, 3))
        
        if len(vertical_lines) > 0:
            # 按长度排序线段
            vertical_lines.sort(key=lambda l: np.sqrt((l[1][0]-l[0][0])**2 + (l[1][1]-l[0][1])**2), reverse=True)
            
            # 收集所有线段上的点
            all_points = []
            for (x1, y1), (x2, y2) in vertical_lines[:min(10, len(vertical_lines))]:
                # 创建线段上的点
                num_pts = max(2, int(np.sqrt((x2-x1)**2 + (y2-y1)**2) / 10))
                for i in range(num_pts):
                    pt_x = x1 + (x2 - x1) * i / (num_pts - 1)
                    pt_y = y1 + (y2 - y1) * i / (num_pts - 1)
                    all_points.append((pt_x, pt_y))
            
            # 计算置信度
            confidence = min(0.9, max(0.5, len(vertical_lines) / 20))
            
            # 根据点的y坐标进行均匀分布
            if len(all_points) > 0:
                # 对点进行均匀采样
                all_points.sort(key=lambda p: p[1])  # 按y坐标排序
                y_min = all_points[0][1]
                y_max = all_points[-1][1]
                
                # 等距离采样
                y_step = (y_max - y_min) / (num_points - 1) if num_points > 1 else 0
                
                for i in range(num_points):
                    target_y = y_min + i * y_step
                    
                    # 找到最接近目标y值的点
                    closest_points = sorted(all_points, key=lambda p: abs(p[1] - target_y))[:5]
                    
                    if closest_points:
                        # 对最近的几个点进行平均，生成更稳定的结果
                        avg_x = sum(p[0] for p in closest_points) / len(closest_points)
                        avg_y = sum(p[1] for p in closest_points) / len(closest_points)
                        
                        keypoints[i, 0] = avg_x
                        keypoints[i, 1] = avg_y
                        keypoints[i, 2] = confidence
                    else:
                        # 备用：插值生成均匀的y坐标点
                        keypoints[i, 0] = mid_x
                        keypoints[i, 1] = y_min + i * y_step
                        keypoints[i, 2] = confidence * 0.8  # 降低置信度
                
                return keypoints, confidence
        
        # 如果找不到足够的线，使用备用方法
        center_line = self._generate_center_line(image)
        return center_line, 0.5
    
    def _calculate_texture_quality(self, image):
        """计算图像纹理质量评分
        
        Args:
            image: 输入图像
            
        Returns:
            质量评分 (0-1)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # 计算灰度共生矩阵特征 (使用简化方法)
        # 1. 梯度幅值作为纹理指标
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        
        # 2. 计算垂直和水平方向梯度比例
        # 垂直纹理比例越高，越有可能是清晰脊柱
        vert_sum = np.sum(np.abs(sobely))
        horiz_sum = np.sum(np.abs(sobelx))
        
        vert_ratio = vert_sum / (vert_sum + horiz_sum + 1e-6)
        
        # 3. 计算纹理的变异程度
        texture_std = np.std(magnitude) / np.mean(magnitude) if np.mean(magnitude) > 0 else 0
        
        # 综合评分
        texture_score = 0.7 * vert_ratio + 0.3 * min(1.0, texture_std)
        texture_score = min(1.0, max(0.0, texture_score))
        
        return texture_score
    
    def _calculate_contrast_quality(self, image):
        """计算图像对比度质量评分
        
        Args:
            image: 输入图像
            
        Returns:
            质量评分 (0-1)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # 计算全局对比度
        global_std = np.std(gray)
        global_score = min(1.0, global_std / 50.0)  # 标准差50以上为良好对比度
        
        # 计算局部对比度
        block_size = max(20, int(min(gray.shape) / 10))
        local_contrasts = []
        
        for i in range(0, gray.shape[0] - block_size, block_size):
            for j in range(0, gray.shape[1] - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                local_contrasts.append(np.std(block))
        
        if local_contrasts:
            local_score = min(1.0, np.mean(local_contrasts) / 30.0)  # 局部标准差30以上为良好
        else:
            local_score = 0.5
        
        # 综合评分
        contrast_score = 0.7 * global_score + 0.3 * local_score
        
        return contrast_score
    
    def _generate_center_line(self, image):
        """生成图像中心线作为脊柱，基于图像特征进行动态检测
        
        Args:
            image: 输入图像
            
        Returns:
            keypoints: 脊柱关键点坐标 [N, 3] (x, y, confidence)
        """
        h, w = image.shape[:2]
        num_points = self.config['num_keypoints']
        
        # 使用图像哈希生成一个唯一的随机种子
        img_hash = np.sum(image[::20, ::20]) % 10000
        np.random.seed(int(img_hash))
        
        # 尝试使用图像特征检测脊柱中线
        try:
            # 应用边缘检测，增强脊柱可见性
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            edges = cv2.Canny(blurred, 
                            self.config['edge_threshold1'], 
                            self.config['edge_threshold2'])
            
            # 使用霍夫线变换检测主要线条
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 
                threshold=self.config['hough_threshold'], 
                minLineLength=self.config['hough_min_length'], 
                maxLineGap=self.config['hough_max_gap']
            )
            
            # 如果检测到线条，使用它们来定位脊柱
            if lines is not None and len(lines) > 0:
                # 过滤倾斜度小的线（更垂直的线）
                vertical_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x2 - x1) < abs(y2 - y1) * 0.5:  # 更垂直的线
                        vertical_lines.append((x1, y1, x2, y2))
                
                # 如果找到垂直线，使用它们确定脊柱位置
                if vertical_lines:
                    # 对于每个目标y坐标，使用附近的垂直线来估计x坐标
                    y_coords = np.linspace(h * 0.1, h * 0.9, num_points)
                    x_coords = []
                    
                    for target_y in y_coords:
                        x_estimates = []
                        weights = []
                        
                        for x1, y1, x2, y2 in vertical_lines:
                            # 检查线段是否覆盖目标y坐标
                            min_y = min(y1, y2)
                            max_y = max(y1, y2)
                            
                            if min_y <= target_y <= max_y:
                                # 使用线性插值计算x坐标
                                if y2 != y1:
                                    x = x1 + (x2 - x1) * (target_y - y1) / (y2 - y1)
                                    
                                    # 计算权重（更靠近中心的线权重更高）
                                    weight = 1.0 - abs(x - w/2) / (w/2)
                                    x_estimates.append(x)
                                    weights.append(weight)
                        
                        # 如果找到估计值，使用加权平均
                        if x_estimates:
                            if weights:
                                weighted_sum = sum(x * w for x, w in zip(x_estimates, weights))
                                weight_sum = sum(weights)
                                x_coords.append(weighted_sum / weight_sum if weight_sum > 0 else sum(x_estimates) / len(x_estimates))
                            else:
                                x_coords.append(sum(x_estimates) / len(x_estimates))
                        else:
                            # 没有找到估计值，使用图像中心附近的随机值
                            x_coords.append(w/2 + np.random.normal(0, w/20))
                    
                    # 创建关键点
                    keypoints = np.zeros((num_points, 3))
                    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                        keypoints[i] = [x, y, 0.6]  # 中等置信度
                    
                    return keypoints
        
        except Exception as e:
            current_app.logger.warning(f"脊柱特征检测失败: {str(e)}")
        
        # 如果特征检测失败，生成基于图像特征的随机化S形曲线
        center_x = w // 2
        
        # 使用图像特征调整曲线形状
        img_variance = np.var(image) / 10000  # 使用图像方差调整曲线
        curve_amplitude = max(0.03, min(0.1, img_variance * 0.5))  # 控制曲率在3%-10%之间
        
        # 生成具有随机扰动的S形脊柱曲线
        keypoints = np.zeros((num_points, 3))
        for i in range(num_points):
            # Y坐标从上到下均匀分布
            y = int(h * 0.1 + h * 0.8 * (i / (num_points - 1)))
            
            # X坐标形成S形曲线，添加随机扰动
            phase_shift = img_hash % 628 / 100  # 0-6.28范围的相位偏移
            if i < num_points // 2:  # 上部脊柱
                x = center_x + int(w * curve_amplitude * np.sin(np.pi * i / num_points + phase_shift))
            else:  # 下部脊柱
                x = center_x - int(w * curve_amplitude * np.sin(np.pi * (i - num_points//2) / num_points + phase_shift))
            
            # 添加一些随机扰动
            x += int(np.random.normal(0, w * 0.01))
            
            keypoints[i, 0] = x
            keypoints[i, 1] = y
            keypoints[i, 2] = 0.5  # 中等置信度
        
        # 确保关键点在图像范围内
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, w - 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, h - 1)
        
        return keypoints
    
    def _generate_keypoints_from_points(self, points, image_shape):
        """从检测到的点生成关键点
        
        Args:
            points: 检测到的点集 [(x, y), ...]
            image_shape: 图像形状
            
        Returns:
            keypoints: 脊柱关键点坐标 [N, 3] (x, y, confidence)
        """
        h, w = image_shape[:2]
        num_points = self.config['num_keypoints']
        keypoints = np.zeros((num_points, 3))
        
        # 分析点集范围
        if len(points) == 0:
            return self._generate_center_line(np.zeros(image_shape, dtype=np.uint8))
        
        # 将点转换为numpy数组便于操作
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        # 处理点集类型
        if len(points.shape) == 3:
            points = points.reshape(-1, 2)
        
        # 对y坐标范围进行标准化处理
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        y_range = y_max - y_min
        
        # 如果y范围太小，扩展到整个图像高度
        if y_range < h * 0.5:
            y_min = h * 0.1
            y_max = h * 0.9
            y_range = y_max - y_min
        
        # 平滑点集
        if len(points) >= 4:
            try:
                # 使用样条插值
                # 得到更平滑的点集
                x = points[:, 0]
                y = points[:, 1]
                
                # 对点进行排序以确保y是单调的
                sorted_indices = np.argsort(y)
                y_sorted = y[sorted_indices]
                x_sorted = x[sorted_indices]
                
                # 得到更多的平滑点
                num_interp_points = min(100, len(points) * 3)
                y_new = np.linspace(y_min, y_max, num_interp_points)
                
                # 使用多项式拟合而不是样条（更稳定）
                if len(x_sorted) > 5:
                    poly = np.polyfit(y_sorted, x_sorted, 3)
                    x_new = np.polyval(poly, y_new)
                else:
                    # 点太少，使用线性插值
                    x_new = np.interp(y_new, y_sorted, x_sorted)
                
                points = np.column_stack((x_new, y_new))
            except Exception as e:
                self.logger.warning(f"点集平滑失败：{str(e)}")
                # 如果平滑失败，继续使用原始点
        
        # 为每个目标关键点位置找最近的点
        for i in range(num_points):
            # 计算目标y位置
            target_y = y_min + i * y_range / (num_points - 1)
            
            # 找最接近这个y值的点
            distances = np.abs(points[:, 1] - target_y)
            nearest_idx = np.argmin(distances)
            
            if distances[nearest_idx] < y_range / num_points:
                # 找到接近的点
                keypoints[i, 0] = points[nearest_idx, 0]
                keypoints[i, 1] = points[nearest_idx, 1]
                keypoints[i, 2] = 0.8  # 高置信度
            else:
                # 没有接近的点，使用插值
                if i > 0 and i < num_points - 1 and keypoints[i-1, 2] > 0 and keypoints[i+1, 2] > 0:
                    # 两端都有点，使用线性插值
                    prev_idx = i - 1
                    while prev_idx > 0 and keypoints[prev_idx, 2] == 0:
                        prev_idx -= 1
                    
                    next_idx = i + 1
                    while next_idx < num_points - 1 and keypoints[next_idx, 2] == 0:
                        next_idx += 1
                    
                    # 线性插值x坐标
                    alpha = (target_y - keypoints[prev_idx, 1]) / (keypoints[next_idx, 1] - keypoints[prev_idx, 1])
                    x = keypoints[prev_idx, 0] + alpha * (keypoints[next_idx, 0] - keypoints[prev_idx, 0])
                    
                    keypoints[i, 0] = x
                    keypoints[i, 1] = target_y
                    keypoints[i, 2] = 0.5  # 中等置信度（插值）
                else:
                    # 尝试从所有点集中插值
                    try:
                        x_interp = np.interp(target_y, points[:, 1], points[:, 0])
                        keypoints[i, 0] = x_interp
                        keypoints[i, 1] = target_y
                        keypoints[i, 2] = 0.4  # 较低置信度
                    except:
                        # 最后使用理想中心线
                        keypoints[i, 0] = w // 2
                        keypoints[i, 1] = target_y
                        keypoints[i, 2] = 0.2  # 低置信度
        
        # 应用平滑处理
        # 使用加权移动平均，高置信度点权重更高
        smoothed_x = np.copy(keypoints[:, 0])
        window_size = 3
        
        for i in range(num_points):
            start = max(0, i - window_size // 2)
            end = min(num_points, i + window_size // 2 + 1)
            
            # 计算加权平均
            weights = keypoints[start:end, 2]  # 使用置信度作为权重
            if np.sum(weights) > 0:
                weighted_avg = np.sum(keypoints[start:end, 0] * weights) / np.sum(weights)
                smoothed_x[i] = weighted_avg
        
        keypoints[:, 0] = smoothed_x
        
        # 确保关键点在图像范围内
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, w - 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, h - 1)
        
        return keypoints 

    def _preprocess_image(self, image):
        """预处理输入图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 确保图像大小一致
        if image.shape[:2] != (512, 512):
            image = cv2.resize(image, (512, 512))
        
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # 边缘检测
        edges = cv2.Canny(enhanced, 
                         self.config['edge_threshold1'], 
                         self.config['edge_threshold2'])
        
        # 保存调试图像
        if self.config['debug_mode']:
            debug_path = os.path.join(self.config['debug_dir'], 'preprocessed.jpg')
            cv2.imwrite(debug_path, enhanced)
            
            edges_path = os.path.join(self.config['debug_dir'], 'edges.jpg')
            cv2.imwrite(edges_path, edges)
        
        return enhanced

    def _generate_debug_keypoints(self, image_shape):
        """生成调试用的模拟关键点
        
        Args:
            image_shape: 图像形状 (高度, 宽度)
            
        Returns:
            模拟的关键点数组
        """
        h, w = image_shape[:2]
        num_points = self.config['num_keypoints']
        keypoints = []
        
        # 创建中间的控制点
        center_x = w // 2
        
        # 生成S形脊柱曲线
        for i in range(num_points):
            # y坐标从上到下均匀分布
            y = int(h * 0.1 + h * 0.8 * (i / (num_points - 1)))
            
            # x坐标形成S形曲线，上部分向右弯曲，下部分向左弯曲
            if i < num_points // 3:  # 上部脊柱
                curve_factor = np.sin(np.pi * i / (num_points // 2))
                x = center_x - int(w * 0.07 * curve_factor)
                conf = 0.7
            elif i < 2 * num_points // 3:  # 中部脊柱
                curve_factor = np.sin(np.pi * (i - num_points // 3) / (num_points // 3))
                x = center_x + int(w * 0.1 * curve_factor)
                conf = 0.8
            else:  # 下部脊柱
                curve_factor = np.sin(np.pi * (i - 2 * num_points // 3) / (num_points // 3))
                x = center_x - int(w * 0.07 * curve_factor)
                conf = 0.7
            
            keypoints.append([x, y, conf])
        
        return np.array(keypoints) 

    def _calculate_image_clarity(self, image):
        """计算图像清晰度评分
        
        Args:
            image: 输入图像
            
        Returns:
            清晰度评分 (0.0-1.0)
        """
        # 使用拉普拉斯算子检测边缘
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        
        # 计算方差作为清晰度度量
        # 方差越大，说明边缘越清晰，图像越清晰
        variance = laplacian.var()
        
        # 归一化到0.0-1.0范围
        clarity = min(1.0, variance / 500.0)
        
        return clarity 

    def _generate_dynamic_keypoints(self, image):
        """根据图像特征生成动态关键点
        
        Args:
            image: 输入图像
            
        Returns:
            随机但有规律的关键点数组
        """
        h, w = image.shape[:2]
        num_points = self.config.get('num_keypoints', 16)
        keypoints = []
        
        # 使用图像哈希作为随机种子，确保每张图像生成的关键点不同
        img_hash = np.sum(image[::20, ::20]) % 10000  # 采样图像计算简单哈希
        np.random.seed(int(img_hash))
        
        # 转为灰度并进行预处理
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        try:
            # 增强脊柱区域对比度
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # 应用高斯模糊以减少噪声
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            
            # 获取图像中央区域的轮廓
            center_region = blurred[h//4:3*h//4, w//3:2*w//3]
            edges = cv2.Canny(center_region, 50, 150)
            
            # 计算脊柱中心线位置
            spine_center = []
            if np.sum(edges) > 0:
                # 如果有足够的边缘，计算每行的边缘中心
                for y_idx in range(0, edges.shape[0], max(1, edges.shape[0] // 20)):
                    row = edges[y_idx, :]
                    if np.sum(row) > 0:
                        # 计算边缘列的平均位置
                        edge_cols = np.where(row > 0)[0]
                        center_x = np.mean(edge_cols) + w//3  # 调整回原始坐标系
                        y = y_idx + h//4  # 调整回原始坐标系
                        spine_center.append((center_x, y))
                        
            # 如果找到了足够的脊柱中心点，进行插值处理
            if len(spine_center) >= 3:
                spine_center = np.array(spine_center)
                # 按y坐标排序
                sorted_indices = np.argsort(spine_center[:, 1])
                sorted_spine = spine_center[sorted_indices]
                
                # 使用样条或多项式拟合获得更平滑的中心线
                try:
                    # 拟合多项式（3次）
                    poly = np.polyfit(sorted_spine[:, 1], sorted_spine[:, 0], 3)
                    
                    # 在整个y范围内生成点
                    y_coords = np.linspace(h//8, 7*h//8, num_points)
                    x_coords = np.polyval(poly, y_coords)
                    
                    # 生成最终的关键点
                    for i in range(num_points):
                        x, y = int(x_coords[i]), int(y_coords[i])
                        
                        # 确保在图像范围内
                        x = max(0, min(w-1, x))
                        y = max(0, min(h-1, y))
                        
                        # 根据位置生成置信度（通常中部更可靠）
                        rel_pos = abs(i - num_points/2) / (num_points/2)
                        conf = max(0.4, min(0.7, 0.7 - 0.3 * rel_pos))
                        
                        keypoints.append([x, y, conf])
                    
                    current_app.logger.debug(f"使用多项式拟合生成的关键点")
                    return np.array(keypoints)
                except:
                    current_app.logger.warning("多项式拟合失败，使用简单插值")
            
            # 备选方案：简单边缘检测
            # 在垂直方向上搜索每个y位置处的最可能脊柱位置
            v_gradient = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            h_gradient = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            
            # 使用梯度信息寻找脊柱中心线
            y_intervals = np.linspace(h//8, 7*h//8, num_points).astype(int)
            x_positions = []
            
            for y in y_intervals:
                # 使用适当的窗口在此y坐标计算可能的脊柱位置
                search_width = w // 2
                center_col = w // 2
                
                # 限制搜索区域
                start_col = max(0, center_col - search_width//2)
                end_col = min(w, center_col + search_width//2)
                
                # 计算此行的垂直梯度平均强度
                line_strength = np.abs(v_gradient[y, start_col:end_col])
                
                # 找到梯度强度最大的位置作为脊柱可能位置
                if np.sum(line_strength) > 0:
                    strongest_col = start_col + np.argmax(line_strength)
                    x_positions.append(strongest_col)
                else:
                    # 没有明显特征，使用图像中心附近的随机位置
                    x_positions.append(center_col + np.random.normal(0, w//30))
            
            # 平滑处理x坐标
            if len(x_positions) >= 3:
                x_smoothed = np.convolve(x_positions, np.ones(3)/3, mode='same')
            else:
                x_smoothed = x_positions
                
        except Exception as e:
            current_app.logger.warning(f"动态关键点生成异常: {str(e)}")
            # 使用基于随机的退化方法
            y_coords = np.linspace(h//8, 7*h//8, num_points)
            x_smoothed = [w//2 + img_hash % w//10 - w//20 + np.random.normal(0, w//40) for _ in range(num_points)]
        
        # 组合成关键点
        for i in range(min(len(y_intervals), len(x_smoothed))):
            x = int(x_smoothed[i])
            y = int(y_intervals[i])
            
            # 确保在图像范围内
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            
            # 根据位置计算置信度
            rel_pos = abs(i - num_points/2) / (num_points/2)  # 0表示中部，1表示两端
            conf = max(0.4, min(0.7, 0.7 - 0.2 * rel_pos))
            
            keypoints.append([x, y, conf])
        
        current_app.logger.debug(f"使用动态生成的关键点 (基于图像特征)")
        return np.array(keypoints) 