import logging
import json
import os
import random
from datetime import datetime
from flask import current_app
from enum import Enum

class SeverityLevel(Enum):
    """侧弯严重程度枚举"""
    MILD = 0       # 轻度（Cobb角10-25度）
    MODERATE = 1   # 中度（Cobb角25-40度）
    SEVERE = 2     # 重度（Cobb角>40度）

class RehabManager:
    """康复训练管理类
    
    负责根据用户的脊柱侧弯情况，生成个性化的康复训练计划，
    并提供康复训练的指导和动作库。
    """
    
    def __init__(self):
        """初始化康复方案管理器"""
        self.logger = logging.getLogger(__name__)
        self.exercises = self._load_exercise_library()
        self.logger.info("康复训练管理器初始化完成")
    
    def _load_exercise_library(self):
        """加载康复训练动作库"""
        exercises = {
            # 轻度脊柱侧弯 (10°-20°) 适用的康复动作
            "mild": [
                {
                    "id": "spine_stretch_basic",
                    "name": "基础脊柱伸展",
                    "type": "spine_stretch",
                    "description": "站立姿势，双脚与肩同宽，双手自然下垂，保持脊柱挺直，颈部延长",
                    "instructions": [
                        "站立，双脚与肩同宽，双臂自然下垂",
                        "挺胸，收腹，保持肩膀放松",
                        "想象头顶被一根线向上拉伸，脊柱自然延长",
                        "保持此姿势15-30秒，正常呼吸"
                    ],
                    "duration": 20,
                    "repetitions": 3,
                    "image_path": "img/rehab/spine_stretch_basic.jpg",
                    "difficulty": 1
                },
                {
                    "id": "pelvic_tilt",
                    "name": "骨盆前后倾斜",
                    "type": "rotation",
                    "description": "站立或躺卧姿势，练习骨盆的前倾和后倾，增强腹肌和下背部肌肉",
                    "instructions": [
                        "站立，双脚与肩同宽，膝盖微屈",
                        "将骨盆向前倾斜（腰部弯曲，臀部向后翘）",
                        "然后将骨盆向后倾斜（腰部平直，臀部向下收）",
                        "缓慢重复此动作，注意感受腹部和背部肌肉的收缩"
                    ],
                    "duration": 15,
                    "repetitions": 10,
                    "image_path": "img/rehab/pelvic_tilt.jpg",
                    "difficulty": 1
                },
                {
                    "id": "cat_cow",
                    "name": "猫牛式伸展",
                    "type": "spine_stretch",
                    "description": "四足跪地姿势，交替进行脊柱的屈伸，增加脊柱灵活性",
                    "instructions": [
                        "四足跪地，手腕在肩下，膝盖在髋下",
                        "吸气，腹部下沉，抬头，脊柱下凹（牛式）",
                        "呼气，拱背，低头，脊柱上凸（猫式）",
                        "缓慢交替进行，配合呼吸"
                    ],
                    "duration": 15,
                    "repetitions": 8,
                    "image_path": "img/rehab/cat_cow.jpg",
                    "difficulty": 2
                },
                {
                    "id": "gentle_side_bend",
                    "name": "轻柔侧弯伸展",
                    "type": "side_bend",
                    "description": "站立姿势，轻柔地向侧面弯曲脊柱，拉伸侧腰肌群",
                    "instructions": [
                        "站立，双脚与肩同宽，右手放在腰上",
                        "左手臂向上伸直，过头顶向右侧弯曲",
                        "感受左侧腰部的轻微拉伸",
                        "保持10-15秒，然后换另一侧"
                    ],
                    "duration": 15,
                    "repetitions": 3,
                    "image_path": "img/rehab/gentle_side_bend.jpg",
                    "difficulty": 1
                }
            ],
            
            # 中度脊柱侧弯 (20°-40°) 适用的康复动作
            "moderate": [
                {
                    "id": "seated_spinal_twist",
                    "name": "坐姿脊柱扭转",
                    "type": "rotation",
                    "description": "坐姿下轻柔扭转脊柱，增强脊柱旋转灵活性",
                    "instructions": [
                        "坐在椅子上或地板上，脊柱保持挺直",
                        "吸气，延长脊柱",
                        "呼气，向右扭转上半身，右手可放在左膝上辅助",
                        "保持扭转5-10秒，然后回到中心",
                        "重复向左侧扭转"
                    ],
                    "duration": 20,
                    "repetitions": 5,
                    "image_path": "img/rehab/seated_spinal_twist.jpg",
                    "difficulty": 2
                },
                {
                    "id": "asymmetric_stretch",
                    "name": "非对称侧弯伸展",
                    "type": "side_bend",
                    "description": "针对凹侧和凸侧进行不同强度的侧弯伸展，平衡肌肉张力",
                    "instructions": [
                        "站立，双脚与肩同宽",
                        "对凸侧（弯曲的外侧）进行较深的侧弯伸展",
                        "对凹侧（弯曲的内侧）进行较轻的侧弯伸展",
                        "每侧保持15-20秒"
                    ],
                    "duration": 20,
                    "repetitions": 3,
                    "image_path": "img/rehab/asymmetric_stretch.jpg",
                    "difficulty": 3
                },
                {
                    "id": "bird_dog",
                    "name": "鸟狗式平衡",
                    "type": "spine_stretch",
                    "description": "四足跪地姿势，对侧手臂和腿同时抬起，增强脊柱稳定性",
                    "instructions": [
                        "四足跪地，手腕在肩下，膝盖在髋下",
                        "保持脊柱中立位，收紧核心肌群",
                        "同时抬起右臂和左腿，伸展至与地面平行",
                        "保持平衡3-5秒，然后换另一侧"
                    ],
                    "duration": 15,
                    "repetitions": 8,
                    "image_path": "img/rehab/bird_dog.jpg",
                    "difficulty": 3
                },
                {
                    "id": "wall_slide",
                    "name": "墙面滑动",
                    "type": "spine_stretch",
                    "description": "背靠墙壁进行上下滑动，训练脊柱对称性和姿势感知",
                    "instructions": [
                        "背部紧贴墙壁站立，脚离墙约一脚长",
                        "确保头部、肩胛骨和骨盆都接触墙面",
                        "保持这种接触，缓慢下蹲至膝盖弯曲约90度",
                        "然后缓慢滑回站立位置"
                    ],
                    "duration": 20,
                    "repetitions": 10,
                    "image_path": "img/rehab/wall_slide.jpg",
                    "difficulty": 2
                }
            ],
            
            # 重度脊柱侧弯 (40°以上) 适用的康复动作
            "severe": [
                {
                    "id": "schroth_method",
                    "name": "施罗特三维呼吸",
                    "type": "spine_stretch",
                    "description": "特定的三维呼吸技术，结合脊柱旋转矫正，专为严重脊柱侧弯设计",
                    "instructions": [
                        "采用根据个人侧弯类型定制的姿势",
                        "向凹陷的胸廓区域定向呼吸",
                        "在呼气时保持矫正姿势",
                        "结合特定的上肢和下肢姿势，提高矫正效果"
                    ],
                    "duration": 30,
                    "repetitions": 5,
                    "image_path": "img/rehab/schroth_method.jpg",
                    "difficulty": 4
                },
                {
                    "id": "modified_side_plank",
                    "name": "改良侧平板支撑",
                    "type": "side_bend",
                    "description": "针对凸侧进行侧平板支撑，增强侧腰肌肉力量",
                    "instructions": [
                        "侧卧在凹侧（弯曲的内侧）",
                        "用前臂和膝盖支撑身体抬起",
                        "保持脊柱在一条直线上",
                        "停留10-30秒，然后换另一侧（但凹侧停留时间更短）"
                    ],
                    "duration": 15,
                    "repetitions": 3,
                    "image_path": "img/rehab/modified_side_plank.jpg",
                    "difficulty": 3
                },
                {
                    "id": "thoracic_mobility",
                    "name": "胸椎活动度训练",
                    "type": "rotation",
                    "description": "针对胸椎区域的旋转活动，改善胸椎活动度",
                    "instructions": [
                        "四足跪地或坐在椅子上",
                        "一只手放在后脑勺",
                        "慢慢向一侧旋转上半身，目视跟随运动",
                        "返回中心位置，然后向另一侧旋转"
                    ],
                    "duration": 15,
                    "repetitions": 5,
                    "image_path": "img/rehab/thoracic_mobility.jpg",
                    "difficulty": 3
                },
                {
                    "id": "core_stability",
                    "name": "核心稳定训练",
                    "type": "spine_stretch",
                    "description": "增强腹部和背部核心肌肉群，提供脊柱更好的支撑",
                    "instructions": [
                        "仰卧，双膝弯曲，双脚平放在地面",
                        "收紧腹部肌肉，将腰部轻轻压向地面",
                        "保持这种收缩状态5-10秒",
                        "放松后重复，逐渐增加保持时间"
                    ],
                    "duration": 10,
                    "repetitions": 10,
                    "image_path": "img/rehab/core_stability.jpg",
                    "difficulty": 2
                }
            ]
        }
        
        # 尝试从文件加载动作库
        try:
            exercise_file = os.path.join(current_app.config.get('MODEL_DIR', ''), 'rehab', 'exercises.json')
            if os.path.exists(exercise_file):
                with open(exercise_file, 'r', encoding='utf-8') as f:
                    loaded_exercises = json.load(f)
                    self.logger.info(f"已加载康复训练动作库: {len(loaded_exercises)} 种难度")
                    return loaded_exercises
        except Exception as e:
            self.logger.warning(f"加载康复训练动作库失败: {str(e)}")
        
        return exercises
    
    def get_severity_level(self, cobb_angle):
        """根据Cobb角度确定侧弯严重程度
        
        Args:
            cobb_angle: Cobb角度值，单位为度
            
        Returns:
            侧弯严重程度枚举
        """
        if cobb_angle < 25:
            return SeverityLevel.MILD
        elif cobb_angle < 40:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.SEVERE
    
    def create_rehab_plan(self, cobb_angle, user_info=None):
        """根据Cobb角度创建个性化康复训练计划
        
        Args:
            cobb_angle: Cobb角度，度数
            user_info: 用户信息字典，可选
            
        Returns:
            plan: 康复训练计划字典
        """
        self.logger.info(f"为Cobb角度 {cobb_angle}° 的用户创建康复训练计划")
        
        # 根据Cobb角度确定侧弯严重程度
        if cobb_angle < 20:
            severity = "mild"
            severity_text = "轻度"
            sessions_per_week = 3
        elif cobb_angle < 40:
            severity = "moderate"
            severity_text = "中度"
            sessions_per_week = 4
        else:
            severity = "severe"
            severity_text = "重度"
            sessions_per_week = 5
        
        # 从动作库中选择合适的训练动作
        available_exercises = self.exercises.get(severity, [])
        
        # 如果该级别没有足够的训练动作，从相邻级别补充
        if len(available_exercises) < 3:
            if severity == "mild" and "moderate" in self.exercises:
                # 如果是轻度，补充中度动作
                available_exercises.extend([e for e in self.exercises["moderate"] if e["difficulty"] <= 2])
            elif severity == "severe" and "moderate" in self.exercises:
                # 如果是重度，补充中度动作
                available_exercises.extend(self.exercises["moderate"])
            
        # 确保有足够的训练动作
        if not available_exercises:
            self.logger.warning(f"未找到适合Cobb角度 {cobb_angle}° 的训练动作")
            # 使用默认动作
            selected_exercises = [
                {
                    "id": "default_stretch",
                    "name": "基础脊柱伸展",
                    "type": "spine_stretch",
                    "description": "站立姿势，保持脊柱挺直",
                    "instructions": ["站立，双脚与肩同宽", "保持脊柱挺直"],
                    "duration": 20,
                    "repetitions": 3,
                    "difficulty": 1
                }
            ]
        else:
            # 随机选择3-5个训练动作
            num_exercises = min(random.randint(3, 5), len(available_exercises))
            selected_exercises = random.sample(available_exercises, num_exercises)
            
            # 按难度排序
            selected_exercises.sort(key=lambda x: x.get("difficulty", 1))
        
        # 创建训练计划
        plan = {
            "plan_id": f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cobb_angle": cobb_angle,
            "severity": severity,
            "severity_text": severity_text,
            "sessions_per_week": sessions_per_week,
            "duration_weeks": 4,
            "exercises": selected_exercises,
            "notes": self._generate_notes(cobb_angle, severity)
        }
        
        # 添加用户信息（如果有）
        if user_info:
            plan["user_info"] = user_info
        
        self.logger.info(f"已创建康复训练计划，包含 {len(selected_exercises)} 个训练动作")
        return plan
    
    def _generate_notes(self, cobb_angle, severity):
        """生成训练计划的注意事项
        
        Args:
            cobb_angle: Cobb角度
            severity: 严重程度
            
        Returns:
            notes: 注意事项列表
        """
        common_notes = [
            "训练时请穿着舒适的衣物，确保活动自由",
            "如果感到疼痛，请立即停止训练并咨询医生",
            "保持规律训练，效果更佳",
            "每次训练前先做5-10分钟的热身活动",
            "训练后可以用热敷缓解肌肉酸痛"
        ]
        
        specific_notes = {
            "mild": [
                "轻度侧弯康复重点在于正确姿势的养成和预防进一步发展",
                "日常生活中注意保持良好坐姿和站姿",
                "可以结合游泳等全身性运动增强躯干肌肉"
            ],
            "moderate": [
                "中度侧弯康复需要更加规律和专注的训练",
                "建议在专业医生或物理治疗师指导下进行训练",
                "注意训练动作的质量，而非数量",
                "可能需要配合支具治疗，请遵医嘱"
            ],
            "severe": [
                "重度侧弯康复必须在专业医疗团队监督下进行",
                "本训练仅作为辅助治疗，不能替代医疗干预",
                "严格遵循医生建议的训练频率和强度",
                "密切观察训练后的反应，及时向医生反馈",
                "可能需要配合支具治疗或考虑手术干预，请遵医嘱"
            ]
        }
        
        notes = common_notes + specific_notes.get(severity, [])
        
        # 根据Cobb角度添加特定建议
        if cobb_angle > 45:
            notes.append("由于您的Cobb角度较大，请务必先咨询脊柱专科医生的意见，确认这些训练适合您的情况")
        
        return notes
    
    def get_exercise_guidance(self, exercise_id):
        """获取特定训练动作的详细指导
        
        Args:
            exercise_id: 训练动作ID
            
        Returns:
            guidance: 训练指导信息字典，如果未找到则返回None
        """
        # 在所有难度级别中查找指定ID的训练动作
        for severity, exercises in self.exercises.items():
            for exercise in exercises:
                if exercise.get("id") == exercise_id:
                    # 添加额外的指导信息
                    guidance = exercise.copy()
                    
                    # 添加注意事项
                    guidance["cautions"] = [
                        "动作应缓慢控制，不要急促",
                        "如感到疼痛，立即停止",
                        "保持正常呼吸，不要屏气"
                    ]
                    
                    # 添加呼吸指导
                    if exercise.get("type") == "spine_stretch":
                        guidance["breathing"] = "伸展时吸气，保持姿势时正常呼吸"
                    elif exercise.get("type") == "side_bend":
                        guidance["breathing"] = "弯曲时呼气，保持姿势时正常呼吸"
                    elif exercise.get("type") == "rotation":
                        guidance["breathing"] = "旋转时呼气，返回中心位置时吸气"
                    
                    return guidance
        
        self.logger.warning(f"未找到ID为 {exercise_id} 的训练动作")
        return None 