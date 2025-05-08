import numpy as np
import math
from pathlib import Path
import json
import time

class PostureAnalyzer:
    """姿态分析器，负责脊柱姿态比对和评估"""
    
    def __init__(self):
        # 脊柱关键点索引（基于MediaPipe姿态模型）
        self.spine_indices = [
            11, 12,  # 肩膀
            23, 24,  # 髋部
            25, 26   # 膝盖
        ]
        
        # 标准姿势模板库
        self.posture_templates = {}
        self.load_templates()
        
    def load_templates(self):
        """加载标准姿势模板"""
        template_dir = Path(__file__).parent / 'templates'
        template_dir.mkdir(exist_ok=True)
        
        template_files = list(template_dir.glob('*.json'))
        if not template_files:
            # 创建默认模板
            self._create_default_templates(template_dir)
            template_files = list(template_dir.glob('*.json'))
            
        for file in template_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    template = json.load(f)
                    self.posture_templates[template['name']] = template
                    print(f"已加载姿势模板: {template['name']}")
            except Exception as e:
                print(f"加载姿势模板失败 {file}: {str(e)}")
                
    def _create_default_templates(self, template_dir):
        """创建默认姿势模板"""
        default_templates = [
            {
                "name": "标准直立姿势",
                "description": "脊柱自然直立，肩膀水平，髋部水平，重心均匀分布",
                "spine_angles": {
                    "lateral_deviation": 0,   # 侧向偏差角度
                    "forward_tilt": 10,       # 前倾角度
                    "shoulder_balance": 0     # 肩部平衡
                },
                "tolerance": 10,  # 允许的角度误差
                "instruction": "请保持自然站立，双脚与肩同宽，重心均匀分布，脊柱垂直于地面"
            },
            {
                "name": "站姿侧屈矫正",
                "description": "站立姿势下的侧屈矫正动作",
                "spine_angles": {
                    "lateral_deviation": 15,  # 特定矫正动作的侧向角度
                    "forward_tilt": 5,
                    "shoulder_balance": 5
                },
                "tolerance": 8,
                "instruction": "站立姿势，上身向右侧弯曲约15度，保持背部挺直，不要前倾或扭转"
            },
            {
                "name": "脊柱前屈伸展",
                "description": "前屈伸展脊柱和腿部后侧肌群",
                "spine_angles": {
                    "lateral_deviation": 0,
                    "forward_tilt": 45,  # 前屈约45度
                    "shoulder_balance": 0
                },
                "tolerance": 10,
                "instruction": "双腿挺直，上身前屈约45度，保持脊柱伸直，不要弯曲膝盖"
            },
            {
                "name": "猫式伸展",
                "description": "四足跪姿下的脊柱伸展，增强脊柱灵活性",
                "spine_angles": {
                    "lateral_deviation": 0,
                    "forward_tilt": -15,  # 脊柱略微向上拱起
                    "shoulder_balance": 0
                },
                "tolerance": 8,
                "instruction": "四足跪姿，双手双膝支撑地面，背部向上拱起，头部微微低垂"
            },
            {
                "name": "髋关节伸展",
                "description": "站姿下的髋关节伸展训练，强化核心和髋部肌群",
                "spine_angles": {
                    "lateral_deviation": 0,
                    "forward_tilt": 5,
                    "shoulder_balance": 0
                },
                "tolerance": 12,
                "instruction": "单腿站立，另一腿向后伸展，保持脊柱中立位，避免骨盆前倾"
            }
        ]
        
        for template in default_templates:
            file_path = template_dir / f"{template['name']}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, ensure_ascii=False, indent=4)
                
    def calculate_spine_angles(self, landmarks):
        """计算脊柱关键角度"""
        # 修复NumPy数组检查
        if landmarks is None or (isinstance(landmarks, np.ndarray) and (landmarks.size == 0 or landmarks.shape[0] < 33)):
            return None
            
        # 适配我们新的关键点格式(数组)
        try:
            if isinstance(landmarks, np.ndarray):
                # 从数组格式转换为字典格式
                landmarks_dict = []
                for i, lm in enumerate(landmarks):
                    landmarks_dict.append({
                        'x': float(lm[0]),
                        'y': float(lm[1]),
                        'z': float(lm[2]) if len(lm) > 2 else 0.0,
                        'visibility': float(lm[3]) if len(lm) > 3 else 1.0
                    })
                landmarks = landmarks_dict
            
            # 提取关键点
            left_shoulder = np.array([landmarks[11]['x'], landmarks[11]['y'], landmarks[11]['z']])
            right_shoulder = np.array([landmarks[12]['x'], landmarks[12]['y'], landmarks[12]['z']])
            left_hip = np.array([landmarks[23]['x'], landmarks[23]['y'], landmarks[23]['z']])
            right_hip = np.array([landmarks[24]['x'], landmarks[24]['y'], landmarks[24]['z']])
            
            # 计算脊柱中线点
            mid_shoulder = (left_shoulder + right_shoulder) / 2
            mid_hip = (left_hip + right_hip) / 2
            
            # 计算脊柱角度
            spine_vector = mid_shoulder - mid_hip
            
            # 侧向偏差（左右倾斜）
            lateral_angle = np.degrees(np.arctan2(spine_vector[0], -spine_vector[1]))
            
            # 前后倾斜（前倾后仰）
            # 计算脊柱向量在YZ平面的投影与Y轴的夹角
            forward_angle = np.degrees(np.arctan2(spine_vector[2], -spine_vector[1]))
            
            # 旋转角度（躯干旋转）
            # 计算肩部向量和髋部向量之间的夹角
            shoulder_vector = right_shoulder - left_shoulder
            hip_vector = right_hip - left_hip
            
            # 归一化向量
            shoulder_vector_norm = np.linalg.norm(shoulder_vector)
            hip_vector_norm = np.linalg.norm(hip_vector)
            
            if shoulder_vector_norm > 0 and hip_vector_norm > 0:
                shoulder_vector = shoulder_vector / shoulder_vector_norm
                hip_vector = hip_vector / hip_vector_norm
                
                # 计算两个向量在XY平面的投影之间的夹角
                shoulder_vector_xy = np.array([shoulder_vector[0], shoulder_vector[1], 0])
                hip_vector_xy = np.array([hip_vector[0], hip_vector[1], 0])
                
                shoulder_vector_xy = shoulder_vector_xy / np.linalg.norm(shoulder_vector_xy)
                hip_vector_xy = hip_vector_xy / np.linalg.norm(hip_vector_xy)
                
                rotation_angle = np.degrees(np.arccos(np.clip(np.dot(shoulder_vector_xy, hip_vector_xy), -1.0, 1.0)))
                
                # 确定旋转方向（顺时针或逆时针）
                cross_product = np.cross(shoulder_vector_xy, hip_vector_xy)
                if cross_product[2] < 0:
                    rotation_angle = -rotation_angle
            else:
                rotation_angle = 0
                
            # 返回计算的角度
            return {
                "lateral_angle": lateral_angle,       # 侧倾角度（左右倾斜）
                "forward_angle": forward_angle,       # 前倾角度（前倾后仰）
                "rotation_angle": rotation_angle,     # 旋转角度（躯干旋转）
                "spine_length": np.linalg.norm(spine_vector)  # 脊柱长度（用于归一化）
            }
            
        except Exception as e:
            print(f"计算脊柱角度错误: {e}")
            return None
    
    def compare_with_template(self, angles, template_name="标准直立姿势"):
        """将当前姿势与模板进行比对"""
        if angles is None:
            return None
            
        template = self.posture_templates.get(template_name)
        if not template:
            # 如果找不到指定模板，尝试使用默认模板
            template = self.posture_templates.get("标准直立姿势")
            if not template:
                return {"error": f"找不到模板: {template_name}"}
            
        # 创建标准模板角度（如果模板中没有角度数据）
        if "spine_angles" not in template:
            template["spine_angles"] = {
                "lateral_angle": 0.0,     # 理想状态下侧倾角度应为0
                "forward_angle": 0.0,     # 理想状态下前倾角度应为0
                "rotation_angle": 0.0     # 理想状态下旋转角度应为0
            }
            
        if "tolerance" not in template:
            template["tolerance"] = 15.0  # 默认容差为15度
            
        target_angles = template["spine_angles"]
        tolerance = template["tolerance"]
        
        # 计算各项指标的偏差 - 使用新的角度格式
        deviations = {
            "lateral_angle": abs(angles.get("lateral_angle", 0) - target_angles.get("lateral_angle", 0)),
            "forward_angle": abs(angles.get("forward_angle", 0) - target_angles.get("forward_angle", 0)),
            "rotation_angle": abs(angles.get("rotation_angle", 0) - target_angles.get("rotation_angle", 0))
        }
        
        # 创建归一化的偏差值 (0-1范围)
        normalized_deviations = {
            k: min(1.0, v / tolerance) for k, v in deviations.items()
        }
        
        # 计算加权分数 (0-100)
        # 给予前倾角度和侧倾角度更高的权重
        weights = {
            "lateral_angle": 0.4,    # 侧倾角度权重
            "forward_angle": 0.4,    # 前倾角度权重
            "rotation_angle": 0.2    # 旋转角度权重
        }
        
        weighted_score = 100 * (1 - sum(normalized_deviations[k] * weights[k] for k in normalized_deviations))
        score = max(0, min(100, weighted_score))  # 确保在0-100范围内
        
        # 生成反馈信息
        feedback = self._generate_feedback(angles, target_angles, deviations, template)
        
        # 确定姿势状态
        if score >= 90:
            status = "优秀"
        elif score >= 75:
            status = "良好"
        elif score >= 60:
            status = "一般"
        else:
            status = "需要改进"
        
        # 获取指导建议
        instruction = template.get("instruction", "请遵循康复指导建议")
        
        return {
            "score": int(score),
            "status": status,
            "feedback": feedback,
            "instruction": instruction,
            "angles": angles,
            "deviations": deviations,
            "normalized_deviations": normalized_deviations
        }
        
    def _generate_feedback(self, current_angles, target_angles, deviations, template=None):
        """根据角度偏差生成反馈建议"""
        feedback = []
        tolerance = template.get("tolerance", 10.0) if template else 10.0
        
        # 轻度、中度、严重偏差的阈值
        slight_threshold = tolerance * 0.5
        moderate_threshold = tolerance * 1.0
        severe_threshold = tolerance * 1.5
        
        # 侧向偏差反馈
        if deviations["lateral_angle"] > slight_threshold:
            if current_angles["lateral_angle"] > target_angles.get("lateral_angle", 0):
                if deviations["lateral_angle"] > severe_threshold:
                    feedback.append("脊柱右侧倾斜过大，请立即向左调整")
                elif deviations["lateral_angle"] > moderate_threshold:
                    feedback.append("脊柱右侧倾斜明显，请向左适当调整")
                else:
                    feedback.append("脊柱略微右侧倾斜，请稍微向左调整")
            else:
                if deviations["lateral_angle"] > severe_threshold:
                    feedback.append("脊柱左侧倾斜过大，请立即向右调整")
                elif deviations["lateral_angle"] > moderate_threshold:
                    feedback.append("脊柱左侧倾斜明显，请向右适当调整")
                else:
                    feedback.append("脊柱略微左侧倾斜，请稍微向右调整")
                
        # 前倾角度反馈
        if deviations["forward_angle"] > slight_threshold:
            if current_angles["forward_angle"] > target_angles.get("forward_angle", 0):
                if deviations["forward_angle"] > severe_threshold:
                    feedback.append("脊柱前倾过大，请立即挺直上身")
                elif deviations["forward_angle"] > moderate_threshold:
                    feedback.append("脊柱前倾明显，请适当挺直上身")
                else:
                    feedback.append("脊柱略微前倾，请稍微挺直上身")
            else:
                if deviations["forward_angle"] > severe_threshold:
                    feedback.append("脊柱后倾过大，请立即适当前倾")
                elif deviations["forward_angle"] > moderate_threshold:
                    feedback.append("脊柱后倾明显，请适当前倾")
                else:
                    feedback.append("脊柱略微后倾，请稍微前倾")
                
        # 旋转角度反馈
        if deviations["rotation_angle"] > slight_threshold:
            if current_angles["rotation_angle"] > target_angles.get("rotation_angle", 0):
                if deviations["rotation_angle"] > severe_threshold:
                    feedback.append("躯干向右旋转过多，请立即调整姿势")
                elif deviations["rotation_angle"] > moderate_threshold:
                    feedback.append("躯干向右旋转明显，请适当调整姿势")
                else:
                    feedback.append("躯干略微向右旋转，请稍微调整姿势")
            else:
                if deviations["rotation_angle"] > severe_threshold:
                    feedback.append("躯干向左旋转过多，请立即调整姿势")
                elif deviations["rotation_angle"] > moderate_threshold:
                    feedback.append("躯干向左旋转明显，请适当调整姿势")
                else:
                    feedback.append("躯干略微向左旋转，请稍微调整姿势")
        
        # 如果偏差很小，给予积极反馈
        if all(d <= slight_threshold for d in deviations.values()):
            if all(d <= slight_threshold/2 for d in deviations.values()):
                feedback.append("姿势非常标准，请保持")
            else:
                feedback.append("姿势基本正确，请继续保持")
                
        # 如果没有生成任何反馈（这种情况不应该发生）
        if not feedback:
            feedback.append("请调整姿势以符合指导要求")
            
        # 添加激励性反馈
        if deviations["lateral_angle"] <= slight_threshold/2 and \
           deviations["forward_angle"] <= slight_threshold/2:
            feedback.append("做得很好，您的躯干位置很稳定")
            
        return feedback 