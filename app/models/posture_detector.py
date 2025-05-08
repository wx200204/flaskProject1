import cv2
import numpy as np
import logging
import os
from pathlib import Path
from flask import current_app
import traceback
import time


class PostureDetector:
    """
    体态评估检测器 - 分析用户从四个角度拍摄的照片，评估体态问题
    支持的体态问题评估：
    - 高低肩
    - 骨盆前倾/后倾
    - 脊柱侧弯
    - 头部前倾
    - 膝盖过伸/内扣/外翻
    - 平/凹/凸足
    """
    
    def __init__(self, config=None):
        """初始化体态评估器
        
        Args:
            config: 配置参数
        """
        self.logger = logging.getLogger(__name__)
        
        # 默认配置
        self.config = {
            'debug_mode': False,  # 调试模式
            'debug_dir': str(Path(__file__).parent.parent.parent / 'debug' / 'posture'),
            'model_confidence_threshold': 0.7,  # 关键点检测置信度阈值
            'shoulder_asymmetry_threshold': 3.0,  # 高低肩标准(度数)
            'pelvic_tilt_threshold': 5.0,      # 骨盆倾斜标准(度数)
            'head_forward_threshold': 10.0,    # 头部前倾标准(度数)
            'knee_angle_normal_range': (170.0, 185.0),  # 膝盖正常角度范围
            'foot_arch_thresholds': {          # 足弓指数阈值
                'flat': 0.28,                  # 高于此值为平足
                'normal': (0.21, 0.28),        # 正常足弓范围
                'high': 0.21                   # 低于此值为高弓足
            }
        }
        
        # 更新配置
        if config:
            self.config.update(config)
            
        # 创建调试目录
        if self.config['debug_mode']:
            os.makedirs(self.config['debug_dir'], exist_ok=True)
    
    def detect_keypoints(self, image):
        """
        检测人体关键点
        
        使用OpenPose模型或其他人体姿态估计模型检测关键点
        
        Args:
            image: 输入图像
            
        Returns:
            关键点数组，包含坐标和置信度
        """
        try:
            # 首先尝试加载OpenPose模型
            if 'current_app' in globals() and hasattr(current_app, 'config'):
                prototxt = current_app.config.get('POSE_MODEL_PATH', {}).get('prototxt')
                caffemodel = current_app.config.get('POSE_MODEL_PATH', {}).get('caffemodel')
                
                if os.path.exists(prototxt) and os.path.exists(caffemodel):
                    # 使用OpenPose COCO模型 (18个关键点)
                    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                    
                    # 预处理图像
                    img_height, img_width = image.shape[:2]
                    # 计算长宽比，保持图像比例调整大小
                    aspect_ratio = img_width / img_height
                    target_height = 368  # OpenPose推荐输入高度
                    target_width = int(aspect_ratio * target_height)
                    target_width -= target_width % 8  # 确保宽度是8的倍数
                    
                    blob = cv2.dnn.blobFromImage(image, 1.0 / 255, 
                                               (target_width, target_height),
                                               (0, 0, 0), swapRB=True, crop=False)
                    
                    # 前向传播
                    net.setInput(blob)
                    output = net.forward()
                    
                    # 处理输出 - COCO模型输出格式
                    num_keypoints = 18  # COCO模型中的关键点数量
                    keypoints = []
                    
                    # COCO关键点顺序:
                    # 0:Nose 1:Neck 2:RShoulder 3:RElbow 4:RWrist 5:LShoulder 6:LElbow 7:LWrist
                    # 8:RHip 9:RKnee 10:RAnkle 11:LHip 12:LKnee 13:LAnkle 14:REye 15:LEye 
                    # 16:REar 17:LEar
                    
                    H = output.shape[2]
                    W = output.shape[3]
                    
                    for i in range(num_keypoints):
                        # 获取当前关键点的置信度图
                        confidence_map = output[0, i, :, :]
                        
                        # 找到最大置信度位置
                        _, conf, _, point = cv2.minMaxLoc(confidence_map)
                        
                        # 将坐标缩放回原始图像
                        x = int((point[0] * img_width) / W)
                        y = int((point[1] * img_height) / H)
                        
                        # 添加关键点 (x, y, confidence)
                        if conf > self.config['model_confidence_threshold']:
                            keypoints.append((x, y, conf))
                        else:
                            keypoints.append((0, 0, 0))  # 低置信度用零坐标表示
                    
                    return keypoints
                else:
                    self.logger.warning("OpenPose模型文件不存在，使用备用方法")
            
            # 备用方法：使用模拟关键点
            # 在实际应用中，推荐使用预训练的人体姿态估计模型
            # 例如MediaPipe、YOLO姿态等
            return self._generate_simulated_keypoints(image)
            
        except Exception as e:
            current_app.logger.error(f"关键点检测失败: {str(e)}")
            current_app.logger.error(traceback.format_exc())
            return self._generate_simulated_keypoints(image)
    
    def analyze_posture(self, front_image, left_image, right_image, back_image):
        """
        分析四面照片评估体态
        
        Args:
            front_image: 正面照片
            left_image: 左侧照片
            right_image: 右侧照片
            back_image: 背面照片
            
        Returns:
            体态评估结果字典
        """
        start_time = time.time()
        
        # 1. 提取四张图片的关键点
        front_keypoints = self.detect_keypoints(front_image)
        left_keypoints = self.detect_keypoints(left_image)
        right_keypoints = self.detect_keypoints(right_image)
        back_keypoints = self.detect_keypoints(back_image)
        
        # 2. 分析正面照片 - 高低肩、骨盆侧倾、脊柱侧弯
        shoulder_analysis = self._analyze_shoulder_asymmetry(front_image, front_keypoints)
        front_analysis = {
            'shoulder_asymmetry': shoulder_analysis,
            'spinal_alignment': self._analyze_spinal_alignment(front_image, front_keypoints),
            'hip_alignment': self._analyze_hip_alignment(front_image, front_keypoints)
        }
        
        # 3. 分析侧面照片 - 头部前倾、骨盆前倾/后倾、膝盖过伸
        side_analysis = {
            'head_forward': self._analyze_head_forward(left_image, left_keypoints),
            'pelvic_tilt': self._analyze_pelvic_tilt(left_image, left_keypoints),
            'knee_hyperextension': self._analyze_knee_alignment(left_image, left_keypoints)
        }
        
        # 如果左右侧数据都可用，取平均值或更可靠的一侧
        if right_keypoints:
            head_forward_right = self._analyze_head_forward(right_image, right_keypoints)
            pelvic_tilt_right = self._analyze_pelvic_tilt(right_image, right_keypoints)
            knee_right = self._analyze_knee_alignment(right_image, right_keypoints)
            
            # 取平均值或置信度更高的结果
            side_analysis['head_forward'] = self._combine_bilateral_results(
                side_analysis['head_forward'], head_forward_right)
            side_analysis['pelvic_tilt'] = self._combine_bilateral_results(
                side_analysis['pelvic_tilt'], pelvic_tilt_right)
            side_analysis['knee_hyperextension'] = self._combine_bilateral_results(
                side_analysis['knee_hyperextension'], knee_right)
        
        # 4. 分析背面照片 - 肩部对称性、脊柱侧弯验证
        back_analysis = {
            'shoulder_symmetry_back': self._analyze_shoulder_asymmetry(back_image, back_keypoints),
            'spinal_alignment_back': self._analyze_spinal_alignment(back_image, back_keypoints),
            'foot_arch': self._analyze_foot_arch(back_image, back_keypoints)
        }
        
        # 5. 整合分析结果
        integrated_result = self._integrate_analysis_results(
            front_analysis, side_analysis, back_analysis)
        
        # 生成报告和可视化结果
        report = self._generate_posture_report(integrated_result)
        visualization = self._create_visualization(
            front_image, left_image, right_image, back_image,
            front_keypoints, left_keypoints, right_keypoints, back_keypoints,
            integrated_result
        )
        
        # 记录处理时间
        processing_time = time.time() - start_time
        self.logger.info(f"体态分析完成，耗时 {processing_time:.2f} 秒")
        
        # 返回结果
        return {
            'integrated_result': integrated_result,
            'report': report,
            'visualization': visualization,
            'processing_time': processing_time
        }
    
    def _analyze_shoulder_asymmetry(self, image, keypoints):
        """分析肩部不对称性（高低肩）
        
        Args:
            image: 图像
            keypoints: 检测到的关键点
            
        Returns:
            肩部分析结果
        """
        # COCO模型中: 2=右肩, 5=左肩
        RIGHT_SHOULDER = 2
        LEFT_SHOULDER = 5
        
        # 检查关键点是否可用
        if (len(keypoints) <= max(RIGHT_SHOULDER, LEFT_SHOULDER) or
            keypoints[RIGHT_SHOULDER][2] < self.config['model_confidence_threshold'] or
            keypoints[LEFT_SHOULDER][2] < self.config['model_confidence_threshold']):
            return {
                'detected': False,
                'message': '未能可靠检测到肩部关键点',
                'angle': 0,
                'deviation': 0,
                'condition': 'unknown'
            }
        
        # 获取肩部坐标
        right_shoulder = keypoints[RIGHT_SHOULDER][:2]
        left_shoulder = keypoints[LEFT_SHOULDER][:2]
        
        # 计算角度(相对于水平线)
        dx = left_shoulder[0] - right_shoulder[0]
        dy = left_shoulder[1] - right_shoulder[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 角度矫正(根据图像角度)
        # 正值表示左肩高于右肩，负值表示右肩高于左肩
        
        # 计算偏差程度
        deviation = abs(angle)
        threshold = self.config['shoulder_asymmetry_threshold']
        
        # 确定高低肩状况
        if deviation < threshold:
            condition = "normal"
            message = "肩部水平，无明显高低肩"
        else:
            if angle > 0:
                condition = "left_high"
                message = f"左肩高于右肩 {deviation:.1f}°"
            else:
                condition = "right_high"
                message = f"右肩高于左肩 {deviation:.1f}°"
        
        # 保存调试图像
        if self.config['debug_mode']:
            debug_img = image.copy()
            cv2.line(debug_img, 
                    (right_shoulder[0], right_shoulder[1]),
                    (left_shoulder[0], left_shoulder[1]),
                    (0, 255, 0), 2)
            cv2.putText(debug_img, f"角度: {angle:.1f}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            debug_path = os.path.join(self.config['debug_dir'], 'shoulder_analysis.jpg')
            cv2.imwrite(debug_path, debug_img)
            
        return {
            'detected': True,
            'message': message,
            'angle': float(angle),
            'deviation': float(deviation),
            'condition': condition,
            'confidence': min(keypoints[RIGHT_SHOULDER][2], keypoints[LEFT_SHOULDER][2])
        }
    
    def _analyze_spinal_alignment(self, image, keypoints):
        """分析脊柱对齐情况
        
        Args:
            image: 图像
            keypoints: 检测到的关键点
            
        Returns:
            脊柱对齐分析结果
        """
        # COCO模型: 1=颈部, 8=右髋, 11=左髋
        NECK = 1
        RIGHT_HIP = 8
        LEFT_HIP = 11
        
        # 检查关键点是否可用
        if (len(keypoints) <= max(NECK, RIGHT_HIP, LEFT_HIP) or
            keypoints[NECK][2] < self.config['model_confidence_threshold'] or
            keypoints[RIGHT_HIP][2] < 0.5 or keypoints[LEFT_HIP][2] < 0.5):
            
            # 尝试使用已有的脊柱检测器
            try:
                from .back_detector import BackSpineDetector
                spine_detector = BackSpineDetector()
                spine_keypoints = spine_detector.detect_spine(image)
                
                if spine_keypoints is not None and len(spine_keypoints) > 5:
                    # 分析脊柱曲线
                    lateral_deviation = self._analyze_spine_curve(spine_keypoints)
                    
                    condition = "normal"
                    message = "脊柱对齐正常"
                    
                    if lateral_deviation > 10:
                        condition = "severe_lateral_curvature"
                        message = f"脊柱有显著侧弯 ({lateral_deviation:.1f}°)"
                    elif lateral_deviation > 5:
                        condition = "moderate_lateral_curvature"
                        message = f"脊柱有中度侧弯 ({lateral_deviation:.1f}°)"
                    
                    return {
                        'detected': True,
                        'message': message,
                        'lateral_deviation': float(lateral_deviation),
                        'condition': condition,
                        'source': 'spine_detector'
                    }
            
            except Exception as e:
                self.logger.warning(f"使用脊柱检测器时出错: {str(e)}")
            
            return {
                'detected': False,
                'message': '未能可靠检测到脊柱关键点',
                'lateral_deviation': 0,
                'condition': 'unknown'
            }
        
        # 获取颈部和髋部中心点
        neck = np.array(keypoints[NECK][:2])
        hip_center = np.array([
            (keypoints[RIGHT_HIP][0] + keypoints[LEFT_HIP][0]) / 2,
            (keypoints[RIGHT_HIP][1] + keypoints[LEFT_HIP][1]) / 2
        ])
        
        # 计算颈部到髋部中心的垂直线偏差
        vertical_angle = np.degrees(np.arctan2(
            neck[0] - hip_center[0],
            hip_center[1] - neck[1]  # Y坐标倒置
        ))
        
        lateral_deviation = abs(vertical_angle)
        
        # 确定脊柱状况
        if lateral_deviation < 3:
            condition = "normal"
            message = "脊柱对齐正常"
        elif lateral_deviation < 7:
            condition = "mild_lateral_shift"
            message = f"脊柱有轻微侧偏 ({lateral_deviation:.1f}°)"
        else:
            condition = "significant_lateral_shift"
            message = f"脊柱有显著侧偏 ({lateral_deviation:.1f}°)"
        
        # 保存调试图像
        if self.config['debug_mode']:
            debug_img = image.copy()
            cv2.line(debug_img, 
                    (int(neck[0]), int(neck[1])),
                    (int(hip_center[0]), int(hip_center[1])),
                    (0, 255, 0), 2)
            cv2.putText(debug_img, f"角度: {vertical_angle:.1f}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            debug_path = os.path.join(self.config['debug_dir'], 'spinal_alignment.jpg')
            cv2.imwrite(debug_path, debug_img)
        
        return {
            'detected': True,
            'message': message,
            'lateral_deviation': float(vertical_angle),
            'condition': condition,
            'confidence': keypoints[NECK][2]
        }
    
    def _analyze_hip_alignment(self, image, keypoints):
        """分析髋部对齐情况
        
        Args:
            image: 图像
            keypoints: 检测到的关键点
            
        Returns:
            髋部对齐分析结果
        """
        # COCO模型: 8=右髋, 11=左髋
        RIGHT_HIP = 8
        LEFT_HIP = 11
        
        # 检查关键点是否可用
        if (len(keypoints) <= max(RIGHT_HIP, LEFT_HIP) or
            keypoints[RIGHT_HIP][2] < self.config['model_confidence_threshold'] or
            keypoints[LEFT_HIP][2] < self.config['model_confidence_threshold']):
            return {
                'detected': False,
                'message': '未能可靠检测到髋部关键点',
                'angle': 0,
                'deviation': 0,
                'condition': 'unknown'
            }
        
        # 获取髋部坐标
        right_hip = keypoints[RIGHT_HIP][:2]
        left_hip = keypoints[LEFT_HIP][:2]
        
        # 计算角度(相对于水平线)
        dx = left_hip[0] - right_hip[0]
        dy = left_hip[1] - right_hip[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 计算偏差程度
        deviation = abs(angle)
        
        # 确定髋部状况
        if deviation < 3:
            condition = "normal"
            message = "髋部水平，无明显不对称"
        else:
            if angle > 0:
                condition = "left_high"
                message = f"左髋高于右髋 {deviation:.1f}°"
            else:
                condition = "right_high"
                message = f"右髋高于左髋 {deviation:.1f}°"
        
        # 保存调试图像
        if self.config['debug_mode']:
            debug_img = image.copy()
            cv2.line(debug_img, 
                    (right_hip[0], right_hip[1]),
                    (left_hip[0], left_hip[1]),
                    (0, 255, 0), 2)
            cv2.putText(debug_img, f"角度: {angle:.1f}°", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            debug_path = os.path.join(self.config['debug_dir'], 'hip_alignment.jpg')
            cv2.imwrite(debug_path, debug_img)
        
        return {
            'detected': True,
            'message': message,
            'angle': float(angle),
            'deviation': float(deviation),
            'condition': condition,
            'confidence': min(keypoints[RIGHT_HIP][2], keypoints[LEFT_HIP][2])
        }
    
    def _analyze_head_forward(self, image, keypoints):
        """分析头部前倾
        
        Args:
            image: 侧面图像
            keypoints: 检测到的关键点
            
        Returns:
            头部前倾分析结果
        """
        # COCO模型: 0=鼻子, 1=颈部, 16或17=耳朵
        NOSE = 0
        NECK = 1
        EAR = 16  # 使用右耳或左耳，取决于侧面方向
        
        # 检查关键点是否可用
        if (len(keypoints) <= max(NOSE, NECK, EAR) or
            keypoints[NECK][2] < self.config['model_confidence_threshold']):
            
            # 尝试使用耳朵和肩膀
            EAR_ALT = 17  # 尝试另一侧耳朵
            SHOULDER = 2  # 右肩，或者5为左肩
            
            if (len(keypoints) > max(EAR_ALT, SHOULDER) and
                keypoints[EAR_ALT][2] >= self.config['model_confidence_threshold'] and
                keypoints[SHOULDER][2] >= self.config['model_confidence_threshold']):
                
                # 使用耳朵和肩膀的位置
                ear = keypoints[EAR_ALT][:2]
                shoulder = keypoints[SHOULDER][:2]
                
                # 计算耳朵相对于肩膀的前倾角度
                dx = ear[0] - shoulder[0]
                dy = ear[1] - shoulder[1]
                forward_angle = np.degrees(np.arctan2(dx, -dy))  # 垂直向上为基准
                
                deviation = forward_angle
                
                if deviation < 5:
                    condition = "normal"
                    message = "头部姿势正常"
                elif deviation < 10:
                    condition = "slight_forward_head"
                    message = f"头部略微前倾 ({deviation:.1f}°)"
                else:
                    condition = "forward_head_posture"
                    message = f"头部明显前倾 ({deviation:.1f}°)"
                
                return {
                    'detected': True,
                    'message': message,
                    'forward_angle': float(forward_angle),
                    'condition': condition,
                    'confidence': keypoints[EAR_ALT][2]
                }
            
            return {
                'detected': False,
                'message': '未能可靠检测到头部关键点',
                'forward_angle': 0,
                'condition': 'unknown'
            }
        
        # 获取鼻子、颈部和耳朵坐标
        nose = np.array(keypoints[NOSE][:2])
        neck = np.array(keypoints[NECK][:2])
        
        # 计算头部前倾角度 (相对于垂直线)
        dx = nose[0] - neck[0]
        dy = nose[1] - neck[1]
        vertical_angle = np.degrees(np.arctan2(dx, -dy))  # 垂直向上为0度
        
        # 计算偏差程度
        deviation = vertical_angle  # 前倾为正值
        threshold = self.config['head_forward_threshold']
        
        # 确定头部状况
        if deviation < 5:
            condition = "normal"
            message = "头部姿势正常"
        elif deviation < threshold:
            condition = "slight_forward_head"
            message = f"头部略微前倾 ({deviation:.1f}°)"
        else:
            condition = "forward_head_posture"
            message = f"头部明显前倾 ({deviation:.1f}°)"
        
        # 保存调试图像
        if self.config['debug_mode']:
            debug_img = image.copy()
            cv2.line(debug_img, 
                    (int(neck[0]), int(neck[1])),
                    (int(neck[0]), int(neck[1] - 100)),  # 垂直参考线
                    (255, 0, 0), 2)
            cv2.line(debug_img, 
                    (int(neck[0]), int(neck[1])),
                    (int(nose[0]), int(nose[1])),
                    (0, 255, 0), 2)
            cv2.putText(debug_img, f"角度: {vertical_angle:.1f}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            debug_path = os.path.join(self.config['debug_dir'], 'head_forward.jpg')
            cv2.imwrite(debug_path, debug_img)
        
        return {
            'detected': True,
            'message': message,
            'forward_angle': float(vertical_angle),
            'condition': condition,
            'confidence': keypoints[NOSE][2]
        }
    
    def _analyze_pelvic_tilt(self, image, keypoints):
        """分析骨盆前倾/后倾
        
        Args:
            image: 侧面图像
            keypoints: 检测到的关键点
            
        Returns:
            骨盆倾斜分析结果
        """
        # 骨盆倾斜分析较为复杂，OpenPose COCO模型没有直接的骨盆关键点
        # 我们可以用髋部和大腿位置估计倾斜
        
        # COCO模型: 8=右髋, 11=左髋, 9=右膝, 12=左膝
        HIP = 8  # 使用右侧或左侧，取决于拍摄角度
        KNEE = 9
        
        # 检查关键点是否可用
        if (len(keypoints) <= max(HIP, KNEE) or
            keypoints[HIP][2] < self.config['model_confidence_threshold'] or
            keypoints[KNEE][2] < self.config['model_confidence_threshold']):
            return {
                'detected': False,
                'message': '未能可靠检测到骨盆位置',
                'tilt_angle': 0,
                'condition': 'unknown'
            }
        
        # 获取髋部和膝盖坐标
        hip = np.array(keypoints[HIP][:2])
        knee = np.array(keypoints[KNEE][:2])
        
        # 计算髋膝连线与垂直线的角度
        dx = knee[0] - hip[0]
        dy = knee[1] - hip[1]
        angle = np.degrees(np.arctan2(dx, dy))  # 垂直向下为0度
        
        # 根据角度确定骨盆倾斜状态
        # 正值表示前倾，负值表示后倾
        tilt_threshold = self.config['pelvic_tilt_threshold']
        
        if abs(angle) < tilt_threshold:
            condition = "normal"
            message = "骨盆位置正常"
        elif angle > 0:
            condition = "anterior_pelvic_tilt"
            message = f"骨盆前倾 ({angle:.1f}°)"
        else:
            condition = "posterior_pelvic_tilt"
            message = f"骨盆后倾 ({abs(angle):.1f}°)"
        
        # 保存调试图像
        if self.config['debug_mode']:
            debug_img = image.copy()
            cv2.line(debug_img, 
                    (int(hip[0]), int(hip[1])),
                    (int(hip[0]), int(hip[1] + 100)),  # 垂直参考线
                    (255, 0, 0), 2)
            cv2.line(debug_img, 
                    (int(hip[0]), int(hip[1])),
                    (int(knee[0]), int(knee[1])),
                    (0, 255, 0), 2)
            cv2.putText(debug_img, f"角度: {angle:.1f}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            debug_path = os.path.join(self.config['debug_dir'], 'pelvic_tilt.jpg')
            cv2.imwrite(debug_path, debug_img)
        
        return {
            'detected': True,
            'message': message,
            'tilt_angle': float(angle),
            'condition': condition,
            'confidence': min(keypoints[HIP][2], keypoints[KNEE][2])
        }
    
    def _analyze_knee_alignment(self, image, keypoints):
        """分析膝盖对齐情况，检测膝盖过伸、内扣或外翻
        
        Args:
            image: 图像
            keypoints: 检测到的关键点
            
        Returns:
            膝盖对齐分析结果
        """
        # COCO模型: 8=右髋, 9=右膝, 10=右踝
        HIP = 8
        KNEE = 9
        ANKLE = 10
        
        # 检查关键点是否可用
        if (len(keypoints) <= max(HIP, KNEE, ANKLE) or
            keypoints[HIP][2] < self.config['model_confidence_threshold'] or
            keypoints[KNEE][2] < self.config['model_confidence_threshold'] or
            keypoints[ANKLE][2] < self.config['model_confidence_threshold']):
            return {
                'detected': False,
                'message': '未能可靠检测到膝盖关键点',
                'knee_angle': 0,
                'condition': 'unknown'
            }
        
        # 获取髋部、膝盖和踝关节坐标
        hip = np.array(keypoints[HIP][:2])
        knee = np.array(keypoints[KNEE][:2])
        ankle = np.array(keypoints[ANKLE][:2])
        
        # 计算膝盖角度
        vector1 = hip - knee
        vector2 = ankle - knee
        
        # 计算两个向量的夹角
        dot_product = np.dot(vector1, vector2)
        norm_v1 = np.linalg.norm(vector1)
        norm_v2 = np.linalg.norm(vector2)
        
        if norm_v1 * norm_v2 == 0:
            cos_angle = 0
        else:
            cos_angle = dot_product / (norm_v1 * norm_v2)
        
        # 限制cos值在[-1, 1]范围内，避免数值误差
        cos_angle = max(-1, min(1, cos_angle))
        knee_angle = np.degrees(np.arccos(cos_angle))
        
        # 根据角度确定膝盖状况
        normal_range = self.config['knee_angle_normal_range']
        
        if normal_range[0] <= knee_angle <= normal_range[1]:
            condition = "normal"
            message = "膝盖角度正常"
        elif knee_angle > normal_range[1]:
            condition = "hyperextended"
            message = f"膝盖过伸 ({knee_angle:.1f}°)"
        else:
            condition = "flexed"
            message = f"膝盖弯曲 ({knee_angle:.1f}°)"
        
        # 保存调试图像
        if self.config['debug_mode']:
            debug_img = image.copy()
            cv2.line(debug_img, 
                    (int(hip[0]), int(hip[1])),
                    (int(knee[0]), int(knee[1])),
                    (0, 255, 0), 2)
            cv2.line(debug_img, 
                    (int(knee[0]), int(knee[1])),
                    (int(ankle[0]), int(ankle[1])),
                    (0, 255, 0), 2)
            cv2.putText(debug_img, f"角度: {knee_angle:.1f}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            debug_path = os.path.join(self.config['debug_dir'], 'knee_alignment.jpg')
            cv2.imwrite(debug_path, debug_img)
        
        return {
            'detected': True,
            'message': message,
            'knee_angle': float(knee_angle),
            'condition': condition,
            'confidence': min(keypoints[HIP][2], keypoints[KNEE][2], keypoints[ANKLE][2])
        }
    
    def _analyze_foot_arch(self, image, keypoints):
        """分析足弓指数
        
        Args:
            image: 图像
            keypoints: 检测到的关键点
            
        Returns:
            足弓指数分析结果
        """
        # 足弓指数分析较为复杂，需要结合图像处理和关键点分析
        # 这里需要实现具体的足弓指数计算逻辑
        # 这里暂时返回一个占位结果
        return {
            'detected': False,
            'message': '足弓指数分析未实现',
            'arch_index': 0,
            'condition': 'unknown'
        }
    
    def _integrate_analysis_results(self, front_analysis, side_analysis, back_analysis):
        """整合分析结果
        
        Args:
            front_analysis: 正面分析结果
            side_analysis: 侧面分析结果
            back_analysis: 背面分析结果
            
        Returns:
            整合后的分析结果
        """
        # 这里需要实现整合逻辑
        # 这里暂时返回一个占位结果
        return {
            'integrated_result': '整合后的分析结果未实现',
            'report': '整合后的报告未实现',
            'visualization': '整合后的可视化结果未实现'
        }
    
    def _generate_posture_report(self, integrated_result):
        """生成体态报告
        
        Args:
            integrated_result: 整合后的分析结果
            
        Returns:
            体态报告
        """
        # 这里需要实现生成报告的逻辑
        # 这里暂时返回一个占位结果
        return '体态报告未实现'
    
    def _create_visualization(self, front_image, left_image, right_image, back_image,
                             front_keypoints, left_keypoints, right_keypoints, back_keypoints,
                             integrated_result):
        """创建体态可视化结果
        
        Args:
            front_image: 正面图像
            left_image: 左侧图像
            right_image: 右侧图像
            back_image: 背面图像
            front_keypoints: 正面关键点
            left_keypoints: 左侧关键点
            right_keypoints: 右侧关键点
            back_keypoints: 背面关键点
            integrated_result: 整合后的分析结果
            
        Returns:
            体态可视化结果
        """
        # 这里需要实现创建可视化的逻辑
        # 这里暂时返回一个占位结果
        return '体态可视化结果未实现'
    
    def _combine_bilateral_results(self, left_result, right_result):
        """合并左右侧分析结果
        
        Args:
            left_result: 左侧分析结果
            right_result: 右侧分析结果
            
        Returns:
            合并后的结果
        """
        # 这里需要实现合并逻辑
        # 这里暂时返回一个占位结果
        return '合并后的结果未实现'
    
    def _generate_simulated_keypoints(self, image):
        """生成模拟的关键点
        
        Args:
            image: 输入图像
            
        Returns:
            模拟的关键点
        """
        # 这里需要实现生成模拟关键点的逻辑
        # 这里暂时返回一个占位结果
        return [(0, 0, 0)] * 18  # 占位，返回18个零点
    
    def _analyze_spine_curve(self, spine_keypoints):
        """分析脊柱曲线
        
        Args:
            spine_keypoints: 脊柱关键点
            
        Returns:
            脊柱曲线分析结果
        """
        # 这里需要实现脊柱曲线分析的逻辑
        # 这里暂时返回一个占位结果
        return 0  # 占位，返回0表示未实现
    
    def _classify_severity(self, value, thresholds):
        """根据值和阈值分类问题严重程度
        
        Args:
            value: 要分类的数值
            thresholds: 严重程度阈值列表 [轻微, 中度, 严重]
            
        Returns:
            严重程度等级 ('normal', 'mild', 'moderate', 'severe')
        """
        if value < thresholds[0]:
            return 'normal'
        elif value < thresholds[1]:
            return 'mild'
        elif value < thresholds[2]:
            return 'severe'
        else:
            return 'very_severe'
    
    def _calculate_angle(self, p1, p2, p3):
        """计算三点形成的角度（度数）
        
        Args:
            p1, p2, p3: 三个点的坐标 [x, y]
            
        Returns:
            三点形成的角度（度数）
        """
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # 确保余弦值在有效范围内
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.degrees(np.arccos(cosine_angle))
        return angle 