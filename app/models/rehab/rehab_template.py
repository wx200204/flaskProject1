import numpy as np
import cv2
import json
import os
import logging
from enum import Enum
from flask import current_app

class RehabTemplateType(Enum):
    """康复动作类型"""
    SPINE_STRETCH = "spine_stretch"  # 脊柱伸展
    SIDE_BEND = "side_bend"         # 侧弯拉伸
    ROTATION = "rotation"           # 旋转动作
    LYING = "lying"                 # 仰卧/侧卧动作
    CUSTOM = "custom"               # 自定义动作

class BodyPart(Enum):
    """人体部位定义"""
    NOSE = 0
    NECK = 1
    RIGHT_SHOULDER = 2
    RIGHT_ELBOW = 3
    RIGHT_WRIST = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    RIGHT_HIP = 8
    RIGHT_KNEE = 9
    RIGHT_ANKLE = 10
    LEFT_HIP = 11
    LEFT_KNEE = 12
    LEFT_ANKLE = 13
    RIGHT_EYE = 14
    LEFT_EYE = 15
    RIGHT_EAR = 16
    LEFT_EAR = 17

class SeverityLevel(Enum):
    """侧弯严重程度枚举"""
    MILD = 0       # 轻度（Cobb角10-25度）
    MODERATE = 1   # 中度（Cobb角25-40度）
    SEVERE = 2     # 重度（Cobb角>40度）

class RehabTemplate:
    """康复动作模板类
    
    定义了标准康复动作的关键点坐标和区域，提供姿势匹配和评分功能
    """
    
    def __init__(self, template_id, name, template_type, severity_level, view_angle="front"):
        """初始化康复动作模板
        
        Args:
            template_id: 模板ID
            name: 模板名称
            template_type: 模板类型（RehabTemplateType）
            severity_level: 适用的侧弯严重程度（SeverityLevel）
            view_angle: 拍摄角度（"front", "side", "back"）
        """
        self.template_id = template_id
        self.name = name
        self.template_type = template_type
        self.severity_level = severity_level
        self.view_angle = view_angle
        
        # 关键点标准坐标 (相对坐标，范围0-1)
        self.keypoints = None
        
        # 关键点容许范围（标准坐标的容许误差范围）
        self.tolerance_areas = {}
        
        # 关节角度标准值
        self.standard_angles = {}
        
        # 关节角度容许范围
        self.angle_tolerances = {}
        
        # 评分权重
        self.scoring_weights = {
            "keypoint_position": 0.5,  # 关键点位置评分权重
            "joint_angle": 0.5,        # 关节角度评分权重
        }
        
        self.logger = logging.getLogger(__name__)
    
    def load_from_json(self, json_data):
        """从JSON数据加载模板
        
        Args:
            json_data: JSON格式的模板数据
            
        Returns:
            success: 是否成功加载
        """
        try:
            self.keypoints = json_data.get("keypoints")
            self.tolerance_areas = json_data.get("tolerance_areas", {})
            self.standard_angles = json_data.get("standard_angles", {})
            self.angle_tolerances = json_data.get("angle_tolerances", {})
            self.scoring_weights = json_data.get("scoring_weights", self.scoring_weights)
            return True
        except Exception as e:
            self.logger.error(f"加载模板数据失败: {str(e)}")
            return False
    
    def save_to_json(self):
        """将模板保存为JSON格式
        
        Returns:
            json_data: JSON格式的模板数据
        """
        return {
            "template_id": self.template_id,
            "name": self.name,
            "template_type": self.template_type.value if isinstance(self.template_type, RehabTemplateType) else self.template_type,
            "severity_level": self.severity_level.value if isinstance(self.severity_level, SeverityLevel) else self.severity_level,
            "view_angle": self.view_angle,
            "keypoints": self.keypoints,
            "tolerance_areas": self.tolerance_areas,
            "standard_angles": self.standard_angles,
            "angle_tolerances": self.angle_tolerances,
            "scoring_weights": self.scoring_weights
        }
    
    def match_pose(self, detected_keypoints, frame_size=None):
        """匹配检测到的姿势与模板
        
        Args:
            detected_keypoints: 检测到的关键点数组
            frame_size: 图像尺寸 (width, height)，用于归一化坐标
            
        Returns:
            score: 匹配得分 (0-100)
            feedback: 姿势反馈
        """
        if self.keypoints is None or detected_keypoints is None:
            return 0, "没有可用的模板或检测到的关键点"
        
        # 归一化检测到的关键点 (如果提供了图像尺寸)
        normalized_keypoints = detected_keypoints
        if frame_size is not None:
            width, height = frame_size
            normalized_keypoints = [[kp[0]/width, kp[1]/height, kp[2]] for kp in detected_keypoints]
        
        # 计算关键点位置匹配得分
        position_score = self._calculate_position_score(normalized_keypoints)
        
        # 计算关节角度匹配得分
        angle_score = self._calculate_angle_score(normalized_keypoints)
        
        # 计算总得分
        total_score = (
            position_score * self.scoring_weights["keypoint_position"] +
            angle_score * self.scoring_weights["joint_angle"]
        ) * 100  # 转换为0-100范围
        
        # 生成反馈
        feedback = self._generate_feedback(normalized_keypoints)
        
        return total_score, feedback
    
    def _calculate_position_score(self, detected_keypoints):
        """计算关键点位置匹配得分 (0-1)
        
        Args:
            detected_keypoints: 检测到的关键点数组
            
        Returns:
            score: 匹配得分 (0-1)
        """
        if not self.keypoints or not detected_keypoints:
            return 0
        
        total_distance = 0
        valid_points = 0
        
        for i, (template_kp, detected_kp) in enumerate(zip(self.keypoints, detected_keypoints)):
            # 跳过低置信度或无效的关键点
            if len(detected_kp) < 3 or detected_kp[2] < 0.5:
                continue
                
            # 计算欧氏距离
            dx = template_kp[0] - detected_kp[0]
            dy = template_kp[1] - detected_kp[1]
            distance = (dx*dx + dy*dy) ** 0.5
            
            # 获取此关键点的容许误差 (默认0.1)
            tolerance = self.tolerance_areas.get(str(i), 0.1)
            
            # 如果距离在容许范围内，则不计入惩罚
            if distance <= tolerance:
                distance = 0
            
            total_distance += distance
            valid_points += 1
        
        if valid_points == 0:
            return 0
            
        # 计算平均距离，转换为得分 (距离越小，得分越高)
        avg_distance = total_distance / valid_points
        score = max(0, 1 - avg_distance * 5)  # 乘以5使得得分更敏感
        
        return score
    
    def _calculate_angle_score(self, detected_keypoints):
        """计算关节角度匹配得分 (0-1)
        
        Args:
            detected_keypoints: 检测到的关键点数组
            
        Returns:
            score: 匹配得分 (0-1)
        """
        if not self.standard_angles or not detected_keypoints:
            return 1  # 没有角度要求时，默认得分为1
        
        total_angle_diff = 0
        valid_angles = 0
        
        # 计算检测到的姿势的关节角度
        detected_angles = self._calculate_joint_angles(detected_keypoints)
        
        for joint, standard_angle in self.standard_angles.items():
            if joint not in detected_angles:
                continue
                
            detected_angle = detected_angles[joint]
            
            # 获取此关节角度的容许误差 (默认10度)
            tolerance = self.angle_tolerances.get(joint, 10)
            
            # 计算角度差异
            angle_diff = abs(standard_angle - detected_angle)
            
            # 如果角度差异在容许范围内，则不计入惩罚
            if angle_diff <= tolerance:
                angle_diff = 0
            
            total_angle_diff += angle_diff
            valid_angles += 1
        
        if valid_angles == 0:
            return 1
            
        # 计算平均角度差异，转换为得分 (差异越小，得分越高)
        avg_angle_diff = total_angle_diff / valid_angles
        score = max(0, 1 - avg_angle_diff / 90)  # 除以90使得得分在0-1范围内
        
        return score
    
    def _calculate_joint_angles(self, keypoints):
        """计算关节角度
        
        Args:
            keypoints: 关键点数组
            
        Returns:
            angles: 关节角度字典
        """
        angles = {}
        
        # 计算颈部-肩膀-肘部角度（双侧）
        # 右侧: NECK(1) - RIGHT_SHOULDER(2) - RIGHT_ELBOW(3)
        angles["right_shoulder"] = self._calculate_angle(
            keypoints[BodyPart.NECK.value], 
            keypoints[BodyPart.RIGHT_SHOULDER.value], 
            keypoints[BodyPart.RIGHT_ELBOW.value]
        )
        
        # 左侧: NECK(1) - LEFT_SHOULDER(5) - LEFT_ELBOW(6)
        angles["left_shoulder"] = self._calculate_angle(
            keypoints[BodyPart.NECK.value], 
            keypoints[BodyPart.LEFT_SHOULDER.value], 
            keypoints[BodyPart.LEFT_ELBOW.value]
        )
        
        # 计算肩膀-髋部-膝盖角度（双侧）
        # 右侧: RIGHT_SHOULDER(2) - RIGHT_HIP(8) - RIGHT_KNEE(9)
        angles["right_hip"] = self._calculate_angle(
            keypoints[BodyPart.RIGHT_SHOULDER.value], 
            keypoints[BodyPart.RIGHT_HIP.value], 
            keypoints[BodyPart.RIGHT_KNEE.value]
        )
        
        # 左侧: LEFT_SHOULDER(5) - LEFT_HIP(11) - LEFT_KNEE(12)
        angles["left_hip"] = self._calculate_angle(
            keypoints[BodyPart.LEFT_SHOULDER.value], 
            keypoints[BodyPart.LEFT_HIP.value], 
            keypoints[BodyPart.LEFT_KNEE.value]
        )
        
        # 计算脊柱角度（颈部-中点-髋部中点）
        neck = keypoints[BodyPart.NECK.value]
        hip_mid = [
            (keypoints[BodyPart.RIGHT_HIP.value][0] + keypoints[BodyPart.LEFT_HIP.value][0]) / 2,
            (keypoints[BodyPart.RIGHT_HIP.value][1] + keypoints[BodyPart.LEFT_HIP.value][1]) / 2,
            0
        ]
        # 计算中点 (假设为颈部和髋部中点的中间点)
        mid_point = [
            (neck[0] + hip_mid[0]) / 2,
            (neck[1] + hip_mid[1]) / 2,
            0
        ]
        
        angles["spine"] = self._calculate_angle(neck, mid_point, hip_mid)
        
        return angles
    
    def _calculate_angle(self, a, b, c):
        """计算由三点形成的角度
        
        Args:
            a: 第一个点 [x, y, ...]
            b: 中间点 [x, y, ...]
            c: 第三个点 [x, y, ...]
            
        Returns:
            angle: 角度 (度)
        """
        if len(a) < 2 or len(b) < 2 or len(c) < 2:
            return 0
        
        # 计算向量
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        
        # 计算向量长度
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        
        if ba_norm == 0 or bc_norm == 0:
            return 0
        
        # 计算点积
        cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
        
        # 防止浮点数精度问题
        cosine_angle = max(-1, min(1, cosine_angle))
        
        # 计算角度（弧度）并转换为度
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def _generate_feedback(self, detected_keypoints):
        """根据姿势匹配结果生成反馈
        
        Args:
            detected_keypoints: 检测到的关键点数组
            
        Returns:
            feedback: 姿势反馈
        """
        feedback = []
        
        # 检查关键点位置
        for i, (template_kp, detected_kp) in enumerate(zip(self.keypoints, detected_keypoints)):
            # 跳过低置信度或无效的关键点
            if len(detected_kp) < 3 or detected_kp[2] < 0.5:
                continue
                
            # 计算距离
            dx = template_kp[0] - detected_kp[0]
            dy = template_kp[1] - detected_kp[1]
            distance = (dx*dx + dy*dy) ** 0.5
            
            # 获取此关键点的容许误差
            tolerance = self.tolerance_areas.get(str(i), 0.1)
            
            # 如果距离超出容许范围，生成反馈
            if distance > tolerance:
                body_part = self._get_body_part_name(i)
                
                # 确定方向
                direction = ""
                if abs(dx) > abs(dy):  # 水平方向偏差更大
                    direction = "向右移动" if dx < 0 else "向左移动"
                else:  # 垂直方向偏差更大
                    direction = "向上移动" if dy < 0 else "向下移动"
                
                feedback.append(f"{body_part}需要{direction}")
        
        # 检查关节角度
        detected_angles = self._calculate_joint_angles(detected_keypoints)
        
        for joint, standard_angle in self.standard_angles.items():
            if joint not in detected_angles:
                continue
                
            detected_angle = detected_angles[joint]
            
            # 获取此关节角度的容许误差
            tolerance = self.angle_tolerances.get(joint, 10)
            
            # 如果角度差异超出容许范围，生成反馈
            angle_diff = standard_angle - detected_angle
            if abs(angle_diff) > tolerance:
                joint_name = self._get_joint_name(joint)
                direction = "弯曲更多" if angle_diff < 0 else "伸展更多"
                
                feedback.append(f"{joint_name}需要{direction}")
        
        if not feedback:
            return "姿势正确，请保持"
            
        return "、".join(feedback)
    
    def _get_body_part_name(self, index):
        """获取人体部位名称
        
        Args:
            index: 关键点索引
            
        Returns:
            name: 人体部位名称
        """
        for part in BodyPart:
            if part.value == index:
                if part == BodyPart.NOSE:
                    return "头部"
                elif part == BodyPart.NECK:
                    return "颈部"
                elif part == BodyPart.RIGHT_SHOULDER:
                    return "右肩"
                elif part == BodyPart.LEFT_SHOULDER:
                    return "左肩"
                elif part == BodyPart.RIGHT_ELBOW:
                    return "右肘"
                elif part == BodyPart.LEFT_ELBOW:
                    return "左肘"
                elif part == BodyPart.RIGHT_WRIST:
                    return "右手腕"
                elif part == BodyPart.LEFT_WRIST:
                    return "左手腕"
                elif part == BodyPart.RIGHT_HIP:
                    return "右髋"
                elif part == BodyPart.LEFT_HIP:
                    return "左髋"
                elif part == BodyPart.RIGHT_KNEE:
                    return "右膝"
                elif part == BodyPart.LEFT_KNEE:
                    return "左膝"
                elif part == BodyPart.RIGHT_ANKLE:
                    return "右踝"
                elif part == BodyPart.LEFT_ANKLE:
                    return "左踝"
                
        return f"关键点{index}"
    
    def _get_joint_name(self, joint_key):
        """获取关节名称
        
        Args:
            joint_key: 关节键名
            
        Returns:
            name: 关节名称
        """
        joint_names = {
            "right_shoulder": "右肩",
            "left_shoulder": "左肩",
            "right_elbow": "右肘",
            "left_elbow": "左肘",
            "right_hip": "右髋",
            "left_hip": "左髋",
            "right_knee": "右膝",
            "left_knee": "左膝",
            "spine": "脊柱"
        }
        
        return joint_names.get(joint_key, joint_key)


class RehabTemplateManager:
    """康复模板管理器
    
    负责加载、存储和管理康复动作模板
    """
    
    def __init__(self):
        """初始化康复模板管理器"""
        self.templates = {}  # 存储所有模板
        self.logger = logging.getLogger(__name__)
        self._load_templates()
    
    def _load_templates(self):
        """加载所有康复模板"""
        try:
            # 尝试从文件加载模板
            template_dir = os.path.join(current_app.config.get('MODEL_DIR', ''), 'rehab', 'templates')
            os.makedirs(template_dir, exist_ok=True)
            
            # 如果目录为空，创建默认模板
            if not os.listdir(template_dir):
                self._create_default_templates(template_dir)
            
            # 加载所有模板文件
            for filename in os.listdir(template_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(template_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                        
                        # 创建模板对象
                        template = RehabTemplate(
                            template_data["template_id"],
                            template_data["name"],
                            template_data["template_type"],
                            template_data["severity_level"],
                            template_data.get("view_angle", "front")
                        )
                        
                        # 加载模板数据
                        if template.load_from_json(template_data):
                            self.templates[template.template_id] = template
            
            self.logger.info(f"成功加载 {len(self.templates)} 个康复模板")
            
        except Exception as e:
            self.logger.error(f"加载康复模板失败: {str(e)}")
    
    def _create_default_templates(self, template_dir):
        """创建默认康复模板"""
        try:
            # 创建基础脊柱伸展模板（正面视角）
            spine_stretch = RehabTemplate(
                "spine_stretch_front_mild",
                "基础脊柱伸展（轻度）",
                RehabTemplateType.SPINE_STRETCH,
                SeverityLevel.MILD,
                "front"
            )
            
            # 设置关键点标准坐标 (18个关键点，归一化到0-1范围)
            spine_stretch.keypoints = [
                [0.5, 0.1, 0.9],  # NOSE
                [0.5, 0.15, 0.9],  # NECK
                [0.45, 0.15, 0.9],  # RIGHT_SHOULDER
                [0.4, 0.3, 0.9],  # RIGHT_ELBOW
                [0.35, 0.45, 0.9],  # RIGHT_WRIST
                [0.55, 0.15, 0.9],  # LEFT_SHOULDER
                [0.6, 0.3, 0.9],  # LEFT_ELBOW
                [0.65, 0.45, 0.9],  # LEFT_WRIST
                [0.47, 0.5, 0.9],  # RIGHT_HIP
                [0.45, 0.7, 0.9],  # RIGHT_KNEE
                [0.45, 0.9, 0.9],  # RIGHT_ANKLE
                [0.53, 0.5, 0.9],  # LEFT_HIP
                [0.55, 0.7, 0.9],  # LEFT_KNEE
                [0.55, 0.9, 0.9],  # LEFT_ANKLE
                [0.48, 0.08, 0.9],  # RIGHT_EYE
                [0.52, 0.08, 0.9],  # LEFT_EYE
                [0.46, 0.09, 0.9],  # RIGHT_EAR
                [0.54, 0.09, 0.9]   # LEFT_EAR
            ]
            
            # 设置关键点容许误差
            spine_stretch.tolerance_areas = {
                "0": 0.05,  # NOSE
                "1": 0.05,  # NECK
                "2": 0.05,  # RIGHT_SHOULDER
                "5": 0.05,  # LEFT_SHOULDER
                "8": 0.05,  # RIGHT_HIP
                "11": 0.05,  # LEFT_HIP
            }
            
            # 设置标准角度
            spine_stretch.standard_angles = {
                "spine": 180,  # 脊柱应该保持挺直
                "right_shoulder": 90,  # 右肩膀角度
                "left_shoulder": 90,  # 左肩膀角度
                "right_hip": 170,  # 右髋角度
                "left_hip": 170,  # 左髋角度
            }
            
            # 设置角度容许误差
            spine_stretch.angle_tolerances = {
                "spine": 15,
                "right_shoulder": 20,
                "left_shoulder": 20,
                "right_hip": 15,
                "left_hip": 15,
            }
            
            # 保存模板
            self._save_template(spine_stretch, template_dir)
            
            # 创建侧弯拉伸模板（侧面视角）
            side_bend = RehabTemplate(
                "side_bend_side_moderate",
                "侧弯拉伸（中度）",
                RehabTemplateType.SIDE_BEND,
                SeverityLevel.MODERATE,
                "side"
            )
            
            # 设置关键点标准坐标 (18个关键点，侧面视图)
            side_bend.keypoints = [
                [0.5, 0.1, 0.9],  # NOSE
                [0.5, 0.15, 0.9],  # NECK
                [0.45, 0.15, 0.9],  # RIGHT_SHOULDER
                [0.45, 0.3, 0.9],  # RIGHT_ELBOW
                [0.45, 0.45, 0.9],  # RIGHT_WRIST
                [0.55, 0.15, 0.9],  # LEFT_SHOULDER (隐藏在侧面)
                [0.55, 0.3, 0.9],  # LEFT_ELBOW (隐藏在侧面)
                [0.55, 0.45, 0.9],  # LEFT_WRIST (隐藏在侧面)
                [0.5, 0.5, 0.9],  # RIGHT_HIP
                [0.5, 0.7, 0.9],  # RIGHT_KNEE
                [0.5, 0.9, 0.9],  # RIGHT_ANKLE
                [0.5, 0.5, 0.9],  # LEFT_HIP (隐藏在侧面)
                [0.5, 0.7, 0.9],  # LEFT_KNEE (隐藏在侧面)
                [0.5, 0.9, 0.9],  # LEFT_ANKLE (隐藏在侧面)
                [0.48, 0.08, 0.9],  # RIGHT_EYE
                [0.52, 0.08, 0.9],  # LEFT_EYE (部分可见)
                [0.46, 0.09, 0.9],  # RIGHT_EAR
                [0.54, 0.09, 0.9]   # LEFT_EAR (隐藏在侧面)
            ]
            
            # 保存侧弯拉伸模板
            self._save_template(side_bend, template_dir)
            
            # 创建仰卧模板
            lying = RehabTemplate(
                "lying_exercise_severe",
                "仰卧康复动作（重度）",
                RehabTemplateType.LYING,
                SeverityLevel.SEVERE,
                "side"
            )
            
            # 设置关键点标准坐标 (仰卧姿势，从侧面观察)
            lying.keypoints = [
                [0.2, 0.3, 0.9],  # NOSE
                [0.25, 0.3, 0.9],  # NECK
                [0.3, 0.3, 0.9],  # RIGHT_SHOULDER
                [0.45, 0.25, 0.9],  # RIGHT_ELBOW
                [0.6, 0.3, 0.9],  # RIGHT_WRIST
                [0.3, 0.3, 0.9],  # LEFT_SHOULDER (隐藏)
                [0.45, 0.25, 0.9],  # LEFT_ELBOW (隐藏)
                [0.6, 0.3, 0.9],  # LEFT_WRIST (隐藏)
                [0.6, 0.5, 0.9],  # RIGHT_HIP
                [0.75, 0.5, 0.9],  # RIGHT_KNEE
                [0.9, 0.5, 0.9],  # RIGHT_ANKLE
                [0.6, 0.5, 0.9],  # LEFT_HIP (隐藏)
                [0.75, 0.5, 0.9],  # LEFT_KNEE (隐藏)
                [0.9, 0.5, 0.9],  # LEFT_ANKLE (隐藏)
                [0.18, 0.29, 0.9],  # RIGHT_EYE
                [0.18, 0.31, 0.9],  # LEFT_EYE (隐藏)
                [0.16, 0.3, 0.9],  # RIGHT_EAR
                [0.16, 0.3, 0.9]   # LEFT_EAR (隐藏)
            ]
            
            # 保存仰卧模板
            self._save_template(lying, template_dir)
            
            self.logger.info("已创建默认康复模板")
            
        except Exception as e:
            self.logger.error(f"创建默认康复模板失败: {str(e)}")
    
    def _save_template(self, template, template_dir):
        """保存康复模板到文件
        
        Args:
            template: 康复模板对象
            template_dir: 模板目录
        """
        try:
            # 将模板转换为JSON
            template_data = template.save_to_json()
            
            # 保存到文件
            file_path = os.path.join(template_dir, f"{template.template_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)
                
            # 添加到模板字典
            self.templates[template.template_id] = template
            
        except Exception as e:
            self.logger.error(f"保存康复模板失败: {str(e)}")
    
    def get_template(self, template_id):
        """获取康复模板
        
        Args:
            template_id: 模板ID
            
        Returns:
            template: 康复模板对象
        """
        return self.templates.get(template_id)
    
    def get_templates_by_type(self, template_type, severity_level=None):
        """根据类型获取康复模板
        
        Args:
            template_type: 模板类型
            severity_level: 严重程度，可选
            
        Returns:
            templates: 符合条件的模板列表
        """
        result = []
        
        for template in self.templates.values():
            if template.template_type == template_type or template.template_type.value == template_type:
                if severity_level is None or template.severity_level == severity_level or template.severity_level.value == severity_level:
                    result.append(template)
        
        return result
    
    def get_templates_by_severity(self, severity_level):
        """根据严重程度获取康复模板
        
        Args:
            severity_level: 严重程度
            
        Returns:
            templates: 符合条件的模板列表
        """
        result = []
        
        for template in self.templates.values():
            if template.severity_level == severity_level or template.severity_level.value == severity_level:
                result.append(template)
        
        return result
    
    def get_all_templates(self):
        """获取所有康复模板
        
        Returns:
            templates: 所有模板的列表
        """
        return list(self.templates.values())
    
    def create_template(self, template_data):
        """创建新的康复模板
        
        Args:
            template_data: 模板数据
            
        Returns:
            template: 创建的模板对象
        """
        try:
            # 创建模板对象
            template = RehabTemplate(
                template_data.get("template_id"),
                template_data.get("name"),
                template_data.get("template_type"),
                template_data.get("severity_level"),
                template_data.get("view_angle", "front")
            )
            
            # 加载模板数据
            if template.load_from_json(template_data):
                # 保存模板
                template_dir = os.path.join(current_app.config.get('MODEL_DIR', ''), 'rehab', 'templates')
                self._save_template(template, template_dir)
                return template
            
            return None
            
        except Exception as e:
            self.logger.error(f"创建康复模板失败: {str(e)}")
            return None
    
    def update_template(self, template_id, template_data):
        """更新康复模板
        
        Args:
            template_id: 模板ID
            template_data: 新的模板数据
            
        Returns:
            success: 是否成功更新
        """
        try:
            # 检查模板是否存在
            if template_id not in self.templates:
                return False
            
            # 更新模板对象
            template = self.templates[template_id]
            
            # 更新属性
            if "name" in template_data:
                template.name = template_data["name"]
            if "template_type" in template_data:
                template.template_type = template_data["template_type"]
            if "severity_level" in template_data:
                template.severity_level = template_data["severity_level"]
            if "view_angle" in template_data:
                template.view_angle = template_data["view_angle"]
            
            # 加载模板数据
            template.load_from_json(template_data)
            
            # 保存更新后的模板
            template_dir = os.path.join(current_app.config.get('MODEL_DIR', ''), 'rehab', 'templates')
            self._save_template(template, template_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新康复模板失败: {str(e)}")
            return False
    
    def delete_template(self, template_id):
        """删除康复模板
        
        Args:
            template_id: 模板ID
            
        Returns:
            success: 是否成功删除
        """
        try:
            # 检查模板是否存在
            if template_id not in self.templates:
                return False
            
            # 删除文件
            template_dir = os.path.join(current_app.config.get('MODEL_DIR', ''), 'rehab', 'templates')
            file_path = os.path.join(template_dir, f"{template_id}.json")
            
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # 从字典中删除
            del self.templates[template_id]
            
            return True
            
        except Exception as e:
            self.logger.error(f"删除康复模板失败: {str(e)}")
            return False 