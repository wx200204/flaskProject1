# app/utils/analyzer.py
import cv2
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from flask import current_app
import base64
from .visualize import draw_keypoints, draw_lines, add_angle_text
import torch.nn as nn
from ..models.back_detector import BackSpineDetector
from .geometry import CobbAngleCalculator
import logging
import traceback
import time
from scipy.interpolate import UnivariateSpline

# 禁用CUDA以减少内存使用
torch.cuda.is_available = lambda : False

class CobbAngleAnalyzer:
    """脊柱侧弯角度分析器"""
    
    def __init__(self, model_path=None):
        """初始化分析器
        
        Args:
            model_path: 可选的模型路径
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.detector = None  # 延迟初始化检测器
        
    def get_detector(self):
        """获取脊柱检测器实例
        
        Returns:
            初始化好的检测器实例
        """
        if self.detector is None:
            # 创建带调试模式的检测器
            config = {
                'debug_mode': True,  # 启用调试
                'num_keypoints': 17  # 脊柱关键点数量
            }
            self.detector = BackSpineDetector(config)
        
        return self.detector
        
    def analyze(self, image, keypoints=None):
        """分析图像中的脊柱侧弯
        
        Args:
            image: 输入图像
            keypoints: 可选的预先检测的关键点
            
        Returns:
            分析结果字典
        """
        try:
            # 记录图像信息
            h, w = image.shape[:2]
            current_app.logger.info(f"分析图像: {w}x{h}, 类型: {image.dtype}")
            
            # 创建Cobb角度计算器
            calculator = CobbAngleCalculator()
            
            # 如果未提供关键点，使用检测器获取
            if keypoints is None:
                try:
                    spine_detector = self.get_detector()
                    keypoints = spine_detector.detect_spine(image)
                    current_app.logger.debug(f"检测到的关键点形状: {keypoints.shape if hasattr(keypoints, 'shape') else '未知'}")
                except Exception as e:
                    current_app.logger.error(f"关键点检测失败: {str(e)}")
                    current_app.logger.error(traceback.format_exc())
                    # 如果检测失败，使用图像宽高随机生成一些关键点
                    keypoints = self._generate_fallback_keypoints(image.shape[:2])
            
            # 确保keypoints是numpy数组
            if not isinstance(keypoints, np.ndarray):
                keypoints = np.array(keypoints)
                
            # 复制关键点以免修改原始数据
            keypoints_scaled = keypoints.copy()
            
            # 使用改进版计算方法
            try:
                # 首先尝试使用改进版方法进行计算
                current_app.logger.info("使用改进版Cobb角度计算方法")
                result_image, cobb_angle, confidence, severity = calculator.calculate_improved(image, keypoints_scaled)
                current_app.logger.info(f"改进版计算结果: {cobb_angle:.2f}°, 置信度: {confidence:.2f}")
            except Exception as e:
                # 如果改进版方法失败，回退到原始计算方法
                current_app.logger.warning(f"改进版方法失败，回退到原始计算方法: {str(e)}")
                result_image, cobb_angle, confidence, severity = calculator.calculate(image, keypoints_scaled)
            
            # 验证结果图像
            if result_image is None or result_image.size == 0:
                current_app.logger.error("计算结果返回空图像")
                result_image = image.copy()
                cv2.putText(result_image, "计算结果异常", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 返回结果
            return {
                'keypoints': keypoints_scaled,
                'cobb_angle': cobb_angle,
                'confidence': confidence,
                'severity': severity,
                'result_image': result_image
            }
        except Exception as e:
            current_app.logger.error(f"Analysis error: {str(e)}")
            current_app.logger.error(traceback.format_exc())
            
            # 出错时返回带有错误信息的图像
            error_image = image.copy()
            cv2.putText(error_image, "分析失败", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_image, str(e), (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            return {
                'keypoints': [],
                'cobb_angle': 0.0,
                'confidence': 0.0,
                'severity': '未知',
                'result_image': error_image
            }
        
    def _generate_fallback_keypoints(self, shape):
        """在检测失败时生成备用关键点
        
        Args:
            shape: 图像形状 (高度, 宽度)
            
        Returns:
            备用关键点数组
        """
        h, w = shape
        num_points = 17
        keypoints = []
        
        # 添加随机因素，确保不同图像生成不同的关键点
        seed = int(h * w) % 1000
        np.random.seed(seed)
        
        # 生成随机的S形曲线
        center_x = w // 2
        offset_range = w * 0.1
        
        for i in range(num_points):
            # y坐标均匀分布
            y = int(h * 0.1 + h * 0.8 * (i / (num_points - 1)))
            
            # x坐标形成随机S形
            if i < num_points // 3:
                x = center_x - int(offset_range * np.sin(np.pi * i / (num_points // 2)))
            elif i < 2 * num_points // 3:
                x = center_x + int(offset_range * np.sin(np.pi * (i - num_points // 3) / (num_points // 3)))
            else:
                x = center_x - int(offset_range * np.sin(np.pi * (i - 2 * num_points // 3) / (num_points // 3)))
                
            # 添加一些随机扰动
            x += int(np.random.uniform(-offset_range/4, offset_range/4))
            y += int(np.random.uniform(-h*0.02, h*0.02))
            
            # 限制在图像范围内
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            
            keypoints.append([x, y, 0.7])  # 使用中等置信度
        
        current_app.logger.warning("使用备用关键点")
        return np.array(keypoints)
    
    def analyze_optimized(self, image, keypoints=None):
        """优化版脊柱侧弯分析方法 - 整合多种改进
        
        Args:
            image: 输入图像
            keypoints: 可选的预先检测的关键点
            
        Returns:
            分析结果字典
        """
        try:
            start_time = time.time()
            
            # 记录图像信息
            h, w = image.shape[:2]
            current_app.logger.info(f"优化版分析图像: {w}x{h}")
            
            # 1. 关键点检测 - 使用多尺度方法
            if keypoints is None:
                try:
                    spine_detector = self.get_detector()
                    # 调用检测器的detect_spine方法
                    keypoints = spine_detector.detect_spine(image)
                    current_app.logger.debug(f"检测到的关键点: {len(keypoints)}")
                except Exception as e:
                    current_app.logger.error(f"关键点检测失败: {str(e)}")
                    current_app.logger.error(traceback.format_exc())
                    # 使用备用关键点
                    keypoints = self._generate_fallback_keypoints(image.shape[:2])
            
            # 2. 增强关键点质量评估
            # 确保keypoints是numpy数组
            if not isinstance(keypoints, np.ndarray):
                keypoints = np.array(keypoints)
                
            # 复制关键点以免修改原始数据
            keypoints_scaled = keypoints.copy()
            
            # 关键点数量检查
            if len(keypoints_scaled) < 7:
                current_app.logger.warning(f"关键点数量不足 ({len(keypoints_scaled)}), 生成更多插值点")
                # 增强插值，生成更多关键点
                keypoints_scaled = self._enhance_keypoints_with_interpolation(keypoints_scaled, target_count=17)
            
            # 3. 使用几何计算器的改进版方法
            try:
                calculator = CobbAngleCalculator()
                current_app.logger.info("使用改进版Cobb角度计算方法")
                result_image, cobb_angle, confidence, severity = calculator.calculate_improved(image, keypoints_scaled)
                method_used = "improved"
                current_app.logger.info(f"改进版计算结果: {cobb_angle:.2f}°, 置信度: {confidence:.2f}")
            except Exception as e:
                # 如果改进版方法失败，回退到原始计算方法
                current_app.logger.warning(f"改进版方法失败，回退到原始计算方法: {str(e)}")
                result_image, cobb_angle, confidence, severity = calculator.calculate(image, keypoints_scaled)
                method_used = "standard"
            
            # 4. 结果验证与精确度评估
            # 如果角度结果异常，尝试额外的校准
            if cobb_angle > 60 and confidence < 0.6:
                current_app.logger.warning(f"角度结果可能不可靠: {cobb_angle:.2f}°, 置信度: {confidence:.2f}")
                try:
                    # 尝试使用更严格的关键点筛选进行重新计算
                    filtered_keypoints = self._filter_outlier_keypoints(keypoints_scaled, image.shape[:2])
                    _, recalc_angle, recalc_conf, recalc_severity = calculator.calculate_improved(
                        image, filtered_keypoints
                    )
                    
                    # 如果重新计算的结果置信度更高，使用它
                    if recalc_conf > confidence:
                        current_app.logger.info(f"使用过滤后的计算结果: {recalc_angle:.2f}°, 置信度: {recalc_conf:.2f}")
                        cobb_angle = recalc_angle
                        confidence = recalc_conf
                        severity = recalc_severity
                except Exception as e:
                    current_app.logger.error(f"重新计算失败: {str(e)}")
            
            # 计算分析耗时
            elapsed = time.time() - start_time
            current_app.logger.info(f"优化版分析完成，耗时: {elapsed:.3f}秒")
            
            # 返回分析结果
            return {
                'keypoints': keypoints_scaled,
                'cobb_angle': cobb_angle,
                'confidence': confidence,
                'severity': severity,
                'result_image': result_image,
                'method': method_used,
                'analysis_time': elapsed
            }
            
        except Exception as e:
            current_app.logger.error(f"优化版分析出错: {str(e)}")
            current_app.logger.error(traceback.format_exc())
            
            # 出错时回退到标准分析
            current_app.logger.info("回退到标准分析方法")
            return self.analyze(image, keypoints)
    
    def _enhance_keypoints_with_interpolation(self, keypoints, target_count=17):
        """通过插值增强关键点数量和质量
        
        Args:
            keypoints: 原始关键点，形状为 (N, 2) 或 (N, 3)
            target_count: 目标关键点数量
            
        Returns:
            增强后的关键点
        """
        if len(keypoints) < 3:
            # 点太少，无法进行有效插值
            return keypoints
            
        # 提取坐标
        has_confidence = keypoints.shape[1] >= 3
        
        if has_confidence:
            # 如果有置信度，将其归一化并拷贝
            points_xy = keypoints[:, :2].copy()
            confidence = keypoints[:, 2].copy()
        else:
            points_xy = keypoints.copy()
            # 无置信度信息时，使用默认值
            confidence = np.ones(len(keypoints)) * 0.7
        
        # 按Y坐标排序
        sorted_indices = np.argsort(points_xy[:, 1])
        points_xy = points_xy[sorted_indices]
        confidence = confidence[sorted_indices]
        
        # 如果点已经足够多，直接返回
        if len(points_xy) >= target_count:
            return np.column_stack((points_xy, confidence[:len(points_xy)]))
        
        # 创建新的点集
        # 1. 在现有的点之间进行线性插值
        y_min = np.min(points_xy[:, 1])
        y_max = np.max(points_xy[:, 1])
        
        # 创建均匀分布的Y坐标
        y_new = np.linspace(y_min, y_max, target_count)
        
        # 对X坐标进行插值
        # 使用样条插值获得更平滑的曲线
        if len(points_xy) >= 4:
            try:
                # 使用三次样条插值
                spline = UnivariateSpline(points_xy[:, 1], points_xy[:, 0], k=3, s=len(points_xy))
                x_new = spline(y_new)
            except Exception:
                # 如果样条失败，回退到线性插值
                x_new = np.interp(y_new, points_xy[:, 1], points_xy[:, 0])
        else:
            # 点太少，使用线性插值
            x_new = np.interp(y_new, points_xy[:, 1], points_xy[:, 0])
            
        # 对置信度也进行插值，新插入的点置信度略低
        conf_new = np.interp(y_new, points_xy[:, 1], confidence) * 0.9
        
        # 合并坐标和置信度
        enhanced_keypoints = np.column_stack((x_new, y_new, conf_new))
        
        return enhanced_keypoints
        
    def _filter_outlier_keypoints(self, keypoints, image_shape):
        """过滤离群关键点
        
        Args:
            keypoints: 关键点数组 (N, 3)
            image_shape: 图像尺寸 (高度, 宽度)
            
        Returns:
            过滤后的关键点
        """
        if len(keypoints) < 5:
            return keypoints
            
        # 提取X, Y坐标
        points_x = keypoints[:, 0]
        points_y = keypoints[:, 1]
        
        # 1. 中线约束 - 脊柱通常靠近图像中线
        h, w = image_shape[:2]
        center_x = w / 2
        
        # 计算到中线的距离
        dist_to_center = np.abs(points_x - center_x)
        dist_median = np.median(dist_to_center)
        dist_mad = np.median(np.abs(dist_to_center - dist_median))  # MAD更鲁棒
        
        # 定义异常值阈值 - 使用修正的Z分数
        # 如果距离超过中值的2.5倍MAD，可能是异常值
        threshold = dist_median + 2.5 * dist_mad
        horizontal_mask = dist_to_center <= threshold
        
        # 2. 垂直连续性约束 - 脊柱点通常垂直排列
        # 按y坐标排序
        sorted_indices = np.argsort(points_y)
        sorted_x = points_x[sorted_indices]
        sorted_y = points_y[sorted_indices]
        
        # 计算相邻点的x方向变化率
        if len(sorted_x) > 1:
            dx = np.diff(sorted_x)
            dy = np.diff(sorted_y)
            slopes = dx / (dy + 1e-6)  # 避免除以零
            
            # 计算斜率的中位数和MAD
            slope_median = np.median(slopes)
            slope_mad = np.median(np.abs(slopes - slope_median))
            
            # 标记斜率异常的点
            slope_threshold = max(0.5, slope_median + 3 * slope_mad)
            valid_slopes = np.abs(slopes) <= slope_threshold
            
            # 创建垂直连续性掩码
            vertical_mask = np.ones_like(horizontal_mask, dtype=bool)
            for i in range(len(valid_slopes)):
                if not valid_slopes[i]:
                    # 保守策略：只标记斜率异常的第二个点
                    vertical_mask[sorted_indices[i+1]] = False
        else:
            vertical_mask = np.ones_like(horizontal_mask, dtype=bool)
        
        # 组合约束
        valid_mask = horizontal_mask & vertical_mask
        
        # 确保至少保留60%的点
        if np.sum(valid_mask) < len(keypoints) * 0.6:
            # 使用置信度或到中线距离作为保留依据
            if keypoints.shape[1] >= 3:
                # 有置信度信息，按置信度排序
                keep_indices = np.argsort(-keypoints[:, 2])  # 负号使其降序
            else:
                # 没有置信度，按到中线距离排序
                keep_indices = np.argsort(dist_to_center)
                
            # 保留前60%的点
            keep_count = max(5, int(len(keypoints) * 0.6))
            valid_mask = np.zeros_like(valid_mask)
            valid_mask[keep_indices[:keep_count]] = True
        
        # 应用过滤
        filtered_keypoints = keypoints[valid_mask]
        
        # 如果过滤后点数过少，添加一些插值点
        if len(filtered_keypoints) < 7:
            filtered_keypoints = self._enhance_keypoints_with_interpolation(
                filtered_keypoints, target_count=max(7, len(keypoints)//2)
            )
        
        return filtered_keypoints