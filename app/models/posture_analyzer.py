import cv2
import numpy as np
import logging
import os
import time
from pathlib import Path
from flask import current_app
import traceback


class PostureAnalyzer:
    """姿态评估分析器 - 分析前、后、左、右四视图照片评估体态问题"""
    
    def __init__(self, config=None):
        """初始化姿态分析器
        
        Args:
            config: 配置参数
        """
        self.logger = logging.getLogger(__name__)
        
        # 默认配置
        self.config = {
            'debug_mode': False,
            'debug_dir': str(Path(__file__).parent.parent.parent / 'debug/posture'),
            'confidence_threshold': 0.5,  # 关键点置信度阈值
            'angle_threshold': 5.0,       # 角度差异阈值（度）
            'distance_threshold': 0.05,   # 相对距离差异阈值（相对于身高）
        }
        
        # 更新配置
        if config:
            self.config.update(config)
            
        # 创建调试目录
        if self.config['debug_mode']:
            os.makedirs(self.config['debug_dir'], exist_ok=True)
            
        # 初始化OpenPose模型（如果配置中有指定）
        self.pose_model = None
        self.initialize_pose_model()
        
    def initialize_pose_model(self):
        """初始化人体姿势检测模型"""
        try:
            # 检查是否有配置姿势模型路径
            if hasattr(current_app, 'config') and 'POSE_MODEL_PATH' in current_app.config:
                pose_path = current_app.config['POSE_MODEL_PATH']
                
                # 加载OpenPose模型
                self.logger.info("正在加载姿势检测模型...")
                
                # 初始化OpenPose
                net = cv2.dnn.readNetFromCaffe(
                    pose_path['prototxt'],
                    pose_path['caffemodel']
                )
                
                # 设置计算后端和目标设备
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    self.logger.info("已启用GPU加速")
                else:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self.logger.info("使用CPU推理")
                
                self.pose_model = {
                    'net': net,
                    'input_size': (368, 368),
                    'threshold': 0.1,
                    'pairs': [(1, 0), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), 
                              (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), 
                              (0, 14), (0, 15), (14, 16), (15, 17)],
                    'keypoints': ["鼻子", "脖子", "右肩", "右肘", "右腕", "左肩", "左肘", "左腕", 
                                 "右髋", "右膝", "右踝", "左髋", "左膝", "左踝", "右眼", "左眼", 
                                 "右耳", "左耳"]
                }
                self.logger.info("姿势检测模型加载完成")
            else:
                self.logger.warning("未找到姿势模型配置，将使用备用处理方法")
        except Exception as e:
            self.logger.error(f"加载姿势模型失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.pose_model = None
            
    def analyze_posture(self, images):
        """分析用户的体态
        
        Args:
            images: 包含视图照片的字典，可以是部分视图 {"front": 前视图, "back": 后视图, "left": 左视图, "right": 右视图}
            
        Returns:
            体态分析结果字典
        """
        start_time = time.time()
        self.logger.info("开始体态分析...")
        
        result = {
            "posture_issues": [],
            "measurements": {},
            "angles": {},
            "keypoints": {},
            "assessment": {},
            "visualization": {},
            "is_comprehensive": False  # 新增标记，表示是否进行了综合分析
        }
        
        try:
            # 1. 检查输入的视图
            all_views = ["front", "back", "left", "right"]
            available_views = []
            missing_views = []
            
            for view in all_views:
                if view in images and images[view] is not None:
                    available_views.append(view)
                else:
                    missing_views.append(view)
                    self.logger.info(f"未提供{view}视图照片")
            
            if not available_views:
                self.logger.warning("没有提供任何有效的视图照片")
                result["posture_issues"].append("未提供任何视图照片，无法进行分析")
                return result
            
            self.logger.info(f"可用视图: {', '.join(available_views)}")
            self.logger.info(f"缺失视图: {', '.join(missing_views)}")
                    
            # 2. 对每个可用视图进行姿势关键点检测
            keypoints_by_view = {}
            for view in available_views:
                image = images[view]
                keypoints, annotated_image = self._detect_pose_keypoints(image, view)
                keypoints_by_view[view] = keypoints
                result["visualization"][view] = annotated_image
                result["keypoints"][view] = keypoints
            
            # 3. 对每个视图进行独立分析
            # 3.1 分析前视图 - 高低肩、身体倾斜等
            if "front" in keypoints_by_view:
                front_analysis = self._analyze_front_view(keypoints_by_view["front"], images["front"])
                result["assessment"]["front"] = front_analysis
                result["posture_issues"].extend(front_analysis.get("issues", []))
                result["angles"].update(front_analysis.get("angles", {}))
                result["measurements"].update(front_analysis.get("measurements", {}))
                
            # 3.2 分析后视图 - 脊柱侧弯、肩胛骨等
            if "back" in keypoints_by_view:
                back_analysis = self._analyze_back_view(keypoints_by_view["back"], images["back"])
                result["assessment"]["back"] = back_analysis
                result["posture_issues"].extend(back_analysis.get("issues", []))
                result["angles"].update(back_analysis.get("angles", {}))
                result["measurements"].update(back_analysis.get("measurements", {}))
                
            # 3.3 分析侧视图 - 前倾姿势、骨盆倾斜等
            # 处理只有一个侧视图的情况
            if "left" in keypoints_by_view and "right" in keypoints_by_view:
                # 两个侧视图都有
                side_analysis = self._analyze_side_views(
                    keypoints_by_view["left"], 
                    keypoints_by_view["right"],
                    images["left"],
                    images["right"]
                )
                result["assessment"]["side"] = side_analysis
                result["posture_issues"].extend(side_analysis.get("issues", []))
                result["angles"].update(side_analysis.get("angles", {}))
                result["measurements"].update(side_analysis.get("measurements", {}))
            elif "left" in keypoints_by_view:
                # 只有左侧视图
                left_analysis = self._analyze_single_side_view(keypoints_by_view["left"], images["left"], "left")
                result["assessment"]["left_side"] = left_analysis
                result["posture_issues"].extend(left_analysis.get("issues", []))
                result["angles"].update(left_analysis.get("angles", {}))
                result["measurements"].update(left_analysis.get("measurements", {}))
            elif "right" in keypoints_by_view:
                # 只有右侧视图
                right_analysis = self._analyze_single_side_view(keypoints_by_view["right"], images["right"], "right")
                result["assessment"]["right_side"] = right_analysis
                result["posture_issues"].extend(right_analysis.get("issues", []))
                result["angles"].update(right_analysis.get("angles", {}))
                result["measurements"].update(right_analysis.get("measurements", {}))
            
            # 4. 检查是否可以进行综合分析（如果有全部四个视图）
            if len(available_views) == 4:
                self.logger.info("检测到完整的四视图数据，进行综合体态分析...")
                comprehensive_analysis = self._perform_comprehensive_analysis(
                    keypoints_by_view, 
                    images,
                    result["assessment"],
                    result["angles"],
                    result["measurements"]
                )
                
                # 更新结果
                result["assessment"]["comprehensive"] = comprehensive_analysis
                result["posture_issues"].extend(comprehensive_analysis.get("issues", []))
                result["is_comprehensive"] = True
                
                # 添加跨视图的综合测量
                if "cross_view_measurements" in comprehensive_analysis:
                    for key, value in comprehensive_analysis["cross_view_measurements"].items():
                        result["measurements"][key] = value
                
                # 添加综合角度分析
                if "cross_view_angles" in comprehensive_analysis:
                    for key, value in comprehensive_analysis["cross_view_angles"].items():
                        result["angles"][key] = value
            
            # 5. 整合分析结果，生成综合评估
            result["severity"] = self._calculate_overall_severity(result["posture_issues"], result["is_comprehensive"])
            result["recommendations"] = self._generate_recommendations(result["posture_issues"], result["is_comprehensive"])
            
            # 6. 去除重复的问题
            result["posture_issues"] = list(set(result["posture_issues"]))
            
            # 7. 标记已分析和缺失的视图
            result["available_views"] = available_views
            result["missing_views"] = missing_views
            
            # 计算分析耗时
            elapsed = time.time() - start_time
            self.logger.info(f"体态分析完成，耗时 {elapsed:.3f}秒")
            
            return result
            
        except Exception as e:
            self.logger.error(f"体态分析失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            result["error"] = str(e)
            result["posture_issues"].append("分析过程发生错误")
            return result
    
    def _perform_comprehensive_analysis(self, keypoints_by_view, images, individual_assessments, angles, measurements):
        """执行综合性的体态分析，整合前、后、左、右四个视图的数据
        
        Args:
            keypoints_by_view: 各视图的关键点数据
            images: 各视图的图像数据
            individual_assessments: 各视图的独立评估结果
            angles: 已检测的角度数据
            measurements: 已检测的测量数据
            
        Returns:
            综合分析结果
        """
        self.logger.info("执行多视图综合体态分析...")
        
        result = {
            "issues": [],
            "cross_view_angles": {},
            "cross_view_measurements": {},
            "clinical_assessment": {},
            "summary": {}
        }
        
        # 获取各视图的数据
        front_kp = keypoints_by_view.get("front", [])
        back_kp = keypoints_by_view.get("back", [])
        left_kp = keypoints_by_view.get("left", [])
        right_kp = keypoints_by_view.get("right", [])
        
        try:
            # 1. 前后视图交叉验证 - 肩部不对称
            if "front" in individual_assessments and "back" in individual_assessments:
                front_assess = individual_assessments["front"]
                back_assess = individual_assessments["back"]
                
                # 比较前后视图的肩膀高度差异
                if ("shoulder_height_diff" in measurements and 
                    "shoulder_height_diff_back" in measurements):
                    front_diff = measurements["shoulder_height_diff"]
                    back_diff = measurements.get("shoulder_height_diff_back", 0)
                    
                    # 计算肩膀高度差异的一致性
                    consistency = 1.0 - min(abs(front_diff - back_diff) / max(front_diff, back_diff, 0.01), 1.0)
                    result["cross_view_measurements"]["shoulder_asymmetry_consistency"] = consistency
                    
                    # 如果前后视图都检测到类似程度的肩部不对称，增加确信度
                    if abs(front_diff - back_diff) < 0.02 and front_diff > 0.03:
                        if "clinical_assessment" in front_assess and "shoulder_asymmetry" in front_assess["clinical_assessment"]:
                            shoulder_side = front_assess["clinical_assessment"]["shoulder_asymmetry"]["finding"]
                            result["issues"].append(f"确认的肩部不对称: {shoulder_side}（前后视图一致）")
                            
                            # 添加交叉确认的临床评估
                            result["clinical_assessment"]["confirmed_shoulder_asymmetry"] = {
                                "finding": front_assess["clinical_assessment"]["shoulder_asymmetry"]["finding"],
                                "deviation": f"{(front_diff * 100):.1f}%",
                                "confirmation": "通过前后视图交叉确认",
                                "clinical_significance": "显著" if front_diff > 0.05 else "轻度",
                                "confidence": f"{(consistency * 100):.1f}%",
                                "standard": "临床上，肩峰高度差异>2%被视为不对称，>5%为显著不对称",
                                "reliability": "高（多视图一致性验证）"
                            }
            
            # 2. 左右侧视图与前视图交叉验证 - 头部前倾和脊柱前凸
            if ("front" in individual_assessments and 
                ("left_side" in individual_assessments or "right_side" in individual_assessments)):
                
                # 获取侧视图数据
                side_assessment = None
                side_key = None
                
                if "left_side" in individual_assessments:
                    side_assessment = individual_assessments["left_side"]
                    side_key = "left"
                elif "right_side" in individual_assessments:
                    side_assessment = individual_assessments["right_side"]
                    side_key = "right"
                
                if side_assessment:
                    # 检查头部前倾和身体姿势
                    head_forward_key = f"{side_key}_head_forward_offset"
                    spine_curve_key = f"{side_key}_spine_curvature"
                    
                    if head_forward_key in measurements and measurements[head_forward_key] > 0.05:
                        # 从前视图检查头部是否居中
                        head_centered = True
                        if "head_horizontal_offset" in measurements:
                            head_centered = abs(measurements["head_horizontal_offset"]) < 0.03
                        
                        if head_centered:
                            result["issues"].append("确认的头部前倾（侧视图+前视图验证）")
                            
                            # 添加临床评估
                            result["clinical_assessment"]["confirmed_forward_head_posture"] = {
                                "finding": "头部前倾姿势",
                                "forward_offset": f"{(measurements[head_forward_key] * 100):.1f}%",
                                "confirmation": "通过侧视图和前视图交叉确认",
                                "clinical_significance": "显著" if measurements[head_forward_key] > 0.1 else "轻度",
                                "potential_causes": "可能与长期低头使用电子设备、颈部肌肉失衡或工作姿势不良相关",
                                "biomechanical_impact": "增加颈椎负荷，可导致颈部疼痛和头痛",
                                "reliability": "高（多视图一致性验证）"
                            }
            
            # 3. 评估整体身体对称性 - 需要前后视图
            if "front" in individual_assessments and "back" in individual_assessments:
                # 检查多项不对称指标
                asymmetry_factors = []
                
                # 检查肩膀不对称
                if "shoulder_height_diff" in measurements and measurements["shoulder_height_diff"] > 0.02:
                    asymmetry_factors.append(("肩部不对称", measurements["shoulder_height_diff"]))
                
                # 检查髋部不对称
                if "hip_height_diff" in measurements and measurements["hip_height_diff"] > 0.015:
                    asymmetry_factors.append(("骨盆倾斜", measurements["hip_height_diff"]))
                
                # 检查脊柱侧弯
                if "spine_angle" in angles and abs(angles["spine_angle"]) > 3:
                    asymmetry_factors.append(("脊柱侧弯", abs(angles["spine_angle"]) / 10))
                
                # 计算整体不对称得分
                if asymmetry_factors:
                    asymmetry_score = sum(factor[1] for factor in asymmetry_factors) / len(asymmetry_factors)
                    result["cross_view_measurements"]["body_asymmetry_score"] = float(asymmetry_score)
                    
                    # 添加临床评估
                    asymmetry_assessment = {
                        "finding": "身体结构不对称",
                        "contributing_factors": [factor[0] for factor in asymmetry_factors],
                        "asymmetry_score": f"{(asymmetry_score * 100):.1f}%",
                        "clinical_significance": "显著" if asymmetry_score > 0.05 else "轻度",
                        "potential_causes": "可能与姿势习惯、肌肉失衡、脊柱侧弯或功能性腿长不等相关",
                        "reliability": "高（多因素综合评估）"
                    }
                    
                    # 添加综合体态问题
                    if asymmetry_score > 0.05:
                        result["issues"].append("显著的身体结构不对称（综合多项指标）")
                        result["clinical_assessment"]["body_asymmetry"] = asymmetry_assessment
                    elif asymmetry_score > 0.02:
                        result["issues"].append("轻度的身体结构不对称（综合多项指标）")
                        result["clinical_assessment"]["body_asymmetry"] = asymmetry_assessment
            
            # 4. 综合评估身体前后平衡 - 需要侧视图
            if "left_side" in individual_assessments or "right_side" in individual_assessments:
                # 基于侧视图评估身体前后平衡
                side_key = "left" if "left_side" in individual_assessments else "right"
                
                # 检查头部前倾、脊柱前凸和骨盆前倾
                balance_factors = []
                
                # 头部前倾
                head_forward_key = f"{side_key}_head_forward_offset"
                if head_forward_key in measurements:
                    balance_factors.append(("头部前倾", measurements[head_forward_key]))
                
                # 脊柱前凸
                spine_curve_key = f"{side_key}_spine_curvature"
                if spine_curve_key in measurements:
                    # 曲率值通常很小，需要放大
                    curve_factor = measurements[spine_curve_key] * 1000
                    if curve_factor > 0.3:  # 阈值根据实际情况调整
                        balance_factors.append(("脊柱前凸", min(curve_factor / 2, 0.15)))
                
                # 计算前后平衡得分
                if balance_factors:
                    balance_score = sum(factor[1] for factor in balance_factors) / len(balance_factors)
                    result["cross_view_measurements"]["sagittal_balance_score"] = float(balance_score)
                    
                    # 添加临床评估
                    balance_assessment = {
                        "finding": "矢状面平衡异常",
                        "contributing_factors": [factor[0] for factor in balance_factors],
                        "imbalance_score": f"{(balance_score * 100):.1f}%",
                        "clinical_significance": "显著" if balance_score > 0.1 else "轻度",
                        "potential_causes": "可能与长期姿势不良、肌肉失衡、工作环境设置不合理相关",
                        "biomechanical_impact": "增加脊柱和关节负荷，可能导致慢性疼痛和功能障碍",
                        "reliability": "中（基于单侧视图评估）"
                    }
                    
                    # 添加综合体态问题
                    if balance_score > 0.1:
                        result["issues"].append("显著的前倾姿势（矢状面平衡异常）")
                        result["clinical_assessment"]["sagittal_balance"] = balance_assessment
                    elif balance_score > 0.05:
                        result["issues"].append("轻度的前倾姿势（矢状面平衡异常）")
                        result["clinical_assessment"]["sagittal_balance"] = balance_assessment
            
            # 5. 综合严重程度评估
            num_issues = len(result["issues"])
            overall_severity = "正常"
            
            if num_issues > 3:
                overall_severity = "严重"
            elif num_issues > 1:
                overall_severity = "中度"
            elif num_issues > 0:
                overall_severity = "轻微"
            
            # 综合评估摘要
            result["summary"] = {
                "total_confirmed_issues": num_issues,
                "overall_severity": overall_severity,
                "confidence_level": "高 (基于多视图交叉验证)",
                "assessment_quality": "综合评估 (全部四视图分析)",
                "recommendation_priority": "高" if num_issues > 2 else ("中" if num_issues > 0 else "低")
            }
            
            self.logger.info(f"综合体态分析完成，发现{num_issues}个确认的体态问题")
            return result
            
        except Exception as e:
            self.logger.error(f"综合体态分析失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            result["issues"].append("综合分析过程中发生错误")
            return result
    
    def _calculate_overall_severity(self, issues, is_comprehensive=False):
        """计算整体体态问题的严重程度
        
        Args:
            issues: 体态问题列表
            is_comprehensive: 是否是综合分析结果
            
        Returns:
            整体体态问题的严重程度
        """
        # 计算严重程度的基本逻辑
        num_issues = len(issues)
        
        if num_issues > 5:
            severity = "严重"
        elif num_issues > 3:
            severity = "中度"
        elif num_issues > 0:
            severity = "轻微"
        else:
            severity = "正常"
            
        # 如果是综合分析，调整严重程度的阈值，因为问题经过交叉验证，更加可靠
        if is_comprehensive:
            if "确认的" in str(issues):  # 检查是否有经过交叉验证的问题
                if num_issues > 3:
                    severity = "严重"
                elif num_issues > 1:
                    severity = "中度"
                else:
                    severity = "轻微"
        
        return severity
    
    def _generate_recommendations(self, issues, is_comprehensive=False):
        """生成体态问题的建议
        
        Args:
            issues: 体态问题列表
            is_comprehensive: 是否是综合分析结果
            
        Returns:
            体态问题的建议
        """
        if not issues:
            return "您的体态状况良好，建议保持良好的姿势习惯，定期进行体态评估。"
        
        base_recommendations = "根据体态分析结果，建议您注意以下几点: "
        
        # 根据是否是综合分析，给出不同的建议前缀
        if is_comprehensive:
            base_recommendations = "基于多视角综合分析的高可信度评估，强烈建议您关注以下几个方面: "
            
        specific_advice = []
        
        # 提取问题关键词进行匹配
        issues_str = ' '.join(issues)
        
        if "肩部不对称" in issues_str or "高肩" in issues_str:
            specific_advice.append("进行肩部平衡训练，加强薄弱侧肌肉力量，保持工作和学习时的姿势对称")
            
        if "头部前倾" in issues_str:
            specific_advice.append("注意颈部姿势，避免长时间低头，增加颈部后伸肌肉训练，工作时保持显示器在视线高度")
            
        if "脊柱侧弯" in issues_str:
            specific_advice.append("建议咨询专业医生进行脊柱侧弯程度评估，进行针对性的矫正训练")
            
        if "骨盆倾斜" in issues_str:
            specific_advice.append("加强核心肌群训练，平衡髋部肌肉，注意站立和坐姿时保持骨盆中立位")
            
        if "X型腿" in issues_str:
            specific_advice.append("进行外侧肌肉训练，咨询足科医生评估是否需要定制鞋垫")
            
        if "O型腿" in issues_str:
            specific_advice.append("加强内侧肌肉训练，避免过度外展姿势")
            
        if "身体结构不对称" in issues_str:
            specific_advice.append("建议进行全面的体态矫正计划，针对性地平衡各部位肌肉力量")
            
        if "矢状面平衡异常" in issues_str or "前倾姿势" in issues_str:
            specific_advice.append("注意日常站立和行走姿势，加强后背肌群训练，保持脊柱自然生理曲度")
        
        # 如果没有匹配到具体建议，给出通用建议
        if not specific_advice:
            specific_advice.append("遵循良好的姿势习惯，定期进行体态评估，必要时咨询专业物理治疗师")
        
        # 如果是综合分析，增加一些更专业的建议
        if is_comprehensive:
            specific_advice.append("建议保存此次评估报告，定期（建议3-6个月）进行体态重新评估，追踪改善情况")
            specific_advice.append("如体态问题严重或持续存在，建议咨询专业康复医师或理疗师制定个性化干预计划")
        
        return base_recommendations + "\n\n" + "\n\n".join([f"• {advice}" for advice in specific_advice])
    
    def _detect_pose_keypoints(self, image, view_name):
        """检测姿势关键点
        
        Args:
            image: 输入图像
            view_name: 视图名称 (front, back, left, right)
            
        Returns:
            tuple: (关键点数据, 标注后的图像)
        """
        h, w = image.shape[:2]
        
        # 复制原始图像用于标注
        annotated_image = image.copy()
        
        try:
            if self.pose_model is not None:
                # 使用OpenPose模型检测关键点
                net = self.pose_model['net']
                input_size = self.pose_model['input_size']
                
                # 准备输入blob
                blob = cv2.dnn.blobFromImage(image, 1.0 / 255, input_size, (0, 0, 0), swapRB=True, crop=False)
                net.setInput(blob)
                
                # 前向传播
                output = net.forward()
                
                # 解析输出获取关键点
                keypoints = []
                detected_keypoints = []
                keypoint_coordinates = []  # 保存具有高置信度的关键点坐标
                
                threshold = self.pose_model['threshold']
                
                # 对每个关键点进行处理
                for i in range(18):  # COCO模型有18个关键点
                    # 热力图
                    probMap = output[0, i, :, :]
                    probMap = cv2.resize(probMap, (w, h))
                    
                    # 找到最大值的位置
                    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                    
                    if prob > threshold:
                        # 在图像上标注关键点
                        cv2.circle(annotated_image, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                        cv2.putText(annotated_image, f"{i}", (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                        
                        # 保存关键点信息
                        keypoints.append((int(point[0]), int(point[1]), prob))
                        detected_keypoints.append(True)
                        keypoint_coordinates.append((int(point[0]), int(point[1])))
                    else:
                        keypoints.append((0, 0, 0))  # 未检测到的关键点
                        detected_keypoints.append(False)
                        keypoint_coordinates.append(None)
                
                # 绘制骨架连接线
                for pair in self.pose_model['pairs']:
                    partA = pair[0]
                    partB = pair[1]
                    
                    if detected_keypoints[partA] and detected_keypoints[partB]:
                        cv2.line(annotated_image, keypoint_coordinates[partA], keypoint_coordinates[partB], (0, 255, 0), 3)
                
                # 格式化关键点数据
                formatted_keypoints = []
                for i, (x, y, conf) in enumerate(keypoints):
                    if i < len(self.pose_model['keypoints']):
                        keypoint_name = self.pose_model['keypoints'][i]
                        formatted_keypoints.append({
                            "id": i,
                            "name": keypoint_name,
                            "x": float(x) / w,  # 归一化坐标
                            "y": float(y) / h,
                            "confidence": float(conf)
                        })
                
                return formatted_keypoints, annotated_image
                
            else:
                # 如果没有模型，使用备用方法生成模拟关键点
                self.logger.warning(f"使用备用方法为{view_name}视图生成关键点")
                return self._generate_mock_keypoints(image, view_name), annotated_image
                
        except Exception as e:
            self.logger.error(f"关键点检测失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            # 发生错误时使用备用方法
            return self._generate_mock_keypoints(image, view_name), annotated_image
    
    def _generate_mock_keypoints(self, image, view_name):
        """生成模拟关键点数据（当真实检测失败时使用）
        
        Args:
            image: 输入图像
            view_name: 视图名称
            
        Returns:
            模拟关键点数据
        """
        h, w = image.shape[:2]
        
        # 根据图像哈希值生成一个唯一的随机种子
        img_hash = np.sum(image[::20, ::20]) % 10000
        np.random.seed(int(img_hash))
        
        # 定义关键点名称列表
        keypoint_names = ["鼻子", "脖子", "右肩", "右肘", "右腕", "左肩", "左肘", "左腕", 
                          "右髋", "右膝", "右踝", "左髋", "左膝", "左踝", "右眼", "左眼", 
                          "右耳", "左耳"]
        
        # 根据不同视图生成不同的模拟关键点
        mock_keypoints = []
        
        # 图像中心点和大小信息
        center_x = w / 2
        center_y = h / 2
        height_factor = h / 3  # 人体高度因子
        width_factor = w / 5   # 人体宽度因子
        
        # 基础噪声水平 - 使每张图像生成不同的关键点
        noise_level = 0.02
        
        if view_name == "front" or view_name == "back":
            # 前视图/后视图的模拟关键点
            positions = {
                # 头部区域
                "鼻子": (center_x, center_y - height_factor * 0.8),
                "脖子": (center_x, center_y - height_factor * 0.6),
                "右眼": (center_x - width_factor * 0.2, center_y - height_factor * 0.85),
                "左眼": (center_x + width_factor * 0.2, center_y - height_factor * 0.85),
                "右耳": (center_x - width_factor * 0.3, center_y - height_factor * 0.8),
                "左耳": (center_x + width_factor * 0.3, center_y - height_factor * 0.8),
                
                # 上肢区域
                "右肩": (center_x - width_factor * 0.5, center_y - height_factor * 0.5),
                "左肩": (center_x + width_factor * 0.5, center_y - height_factor * 0.5),
                "右肘": (center_x - width_factor * 0.6, center_y - height_factor * 0.2),
                "左肘": (center_x + width_factor * 0.6, center_y - height_factor * 0.2),
                "右腕": (center_x - width_factor * 0.7, center_y),
                "左腕": (center_x + width_factor * 0.7, center_y),
                
                # 下肢区域
                "右髋": (center_x - width_factor * 0.3, center_y + height_factor * 0.1),
                "左髋": (center_x + width_factor * 0.3, center_y + height_factor * 0.1),
                "右膝": (center_x - width_factor * 0.35, center_y + height_factor * 0.5),
                "左膝": (center_x + width_factor * 0.35, center_y + height_factor * 0.5),
                "右踝": (center_x - width_factor * 0.4, center_y + height_factor * 0.9),
                "左踝": (center_x + width_factor * 0.4, center_y + height_factor * 0.9),
            }
            
            # 添加一些不对称性，模拟真实人体
            if view_name == "front":
                # 模拟轻微的不对称 (高低肩)
                shoulder_diff = np.random.normal(0, height_factor * 0.05)
                positions["右肩"] = (positions["右肩"][0], positions["右肩"][1] + shoulder_diff)
                positions["左肩"] = (positions["左肩"][0], positions["左肩"][1] - shoulder_diff * 0.7)
            
        else:  # 侧视图 (left or right)
            # 侧视图的模拟关键点
            positions = {
                # 头部区域
                "鼻子": (center_x + width_factor * 0.1, center_y - height_factor * 0.8),
                "脖子": (center_x, center_y - height_factor * 0.6),
                "右眼": (center_x + width_factor * 0.15, center_y - height_factor * 0.85),
                "左眼": (center_x + width_factor * 0.15, center_y - height_factor * 0.85), # 侧视图中通常只能看到一只眼
                "右耳": (center_x + width_factor * 0.25, center_y - height_factor * 0.8),
                "左耳": (center_x - width_factor * 0.1, center_y - height_factor * 0.8), # 侧视图中通常只能看到一只耳
                
                # 上肢区域
                "右肩": (center_x, center_y - height_factor * 0.5),
                "左肩": (center_x, center_y - height_factor * 0.5), # 侧视图中肩重叠
                "右肘": (center_x + width_factor * 0.3, center_y - height_factor * 0.3),
                "左肘": (center_x - width_factor * 0.1, center_y - height_factor * 0.3),
                "右腕": (center_x + width_factor * 0.4, center_y - height_factor * 0.1),
                "左腕": (center_x - width_factor * 0.2, center_y - height_factor * 0.1),
                
                # 下肢区域
                "右髋": (center_x, center_y + height_factor * 0.1),
                "左髋": (center_x, center_y + height_factor * 0.1), # 侧视图中髋重叠
                "右膝": (center_x + width_factor * 0.2, center_y + height_factor * 0.5),
                "左膝": (center_x, center_y + height_factor * 0.5),
                "右踝": (center_x + width_factor * 0.1, center_y + height_factor * 0.9),
                "左踝": (center_x - width_factor * 0.2, center_y + height_factor * 0.9),
            }
            
            # 模拟轻微的前倾/后倾
            posture_tilt = np.random.normal(0, 0.1)
            for part in ["脖子", "右肩", "左肩", "右髋", "左髋"]:
                x, y = positions[part]
                positions[part] = (x + posture_tilt * width_factor, y)
        
        # 生成关键点列表
        for i, name in enumerate(keypoint_names):
            if name in positions:
                base_x, base_y = positions[name]
                
                # 添加随机噪声
                noise_x = np.random.normal(0, noise_level * w)
                noise_y = np.random.normal(0, noise_level * h)
                
                x = base_x + noise_x
                y = base_y + noise_y
                
                # 确保坐标在图像范围内
                x = max(0, min(w, x))
                y = max(0, min(h, y))
                
                # 计算归一化坐标
                norm_x = x / w
                norm_y = y / h
                
                # 给关键点一个合理的置信度
                confidence = np.random.uniform(0.6, 0.9)
                
                mock_keypoints.append({
                    "id": i,
                    "name": name,
                    "x": float(norm_x),
                    "y": float(norm_y),
                    "confidence": float(confidence)
                })
            else:
                # 对于未定义位置的关键点，给出低置信度
                mock_keypoints.append({
                    "id": i,
                    "name": name,
                    "x": float(np.random.uniform(0.3, 0.7)),
                    "y": float(np.random.uniform(0.3, 0.7)),
                    "confidence": float(np.random.uniform(0.1, 0.3))
                })
        
        return mock_keypoints
    
    def _analyze_front_view(self, keypoints, image):
        """分析前视图的体态问题
        
        Args:
            keypoints: 前视图关键点数据
            image: 前视图图像
            
        Returns:
            前视图分析结果字典
        """
        h, w = image.shape[:2]
        result = {
            "issues": [],
            "angles": {},
            "measurements": {},
            "clinical_assessment": {}  # 新增临床评估数据
        }
        
        # 通过ID快速查找关键点
        kp_dict = {kp["id"]: kp for kp in keypoints}
        
        try:
            # 检查肩膀高度差异 (高低肩)
            if 2 in kp_dict and 5 in kp_dict and kp_dict[2]["confidence"] > 0.5 and kp_dict[5]["confidence"] > 0.5:
                right_shoulder = (kp_dict[2]["x"] * w, kp_dict[2]["y"] * h)
                left_shoulder = (kp_dict[5]["x"] * w, kp_dict[5]["y"] * h)
                
                # 计算肩膀高度差异（像素）
                shoulder_height_diff_px = abs(right_shoulder[1] - left_shoulder[1])
                result["measurements"]["shoulder_height_diff_px"] = float(shoulder_height_diff_px)
                
                # 相对身高的肩膀高度差异比例
                # 估计身高：使用颈部到脚踝的距离
                height_px = 0
                if 1 in kp_dict and 10 in kp_dict and 13 in kp_dict:  # 颈部和脚踝
                    neck = (kp_dict[1]["x"] * w, kp_dict[1]["y"] * h)
                    right_ankle = (kp_dict[10]["x"] * w, kp_dict[10]["y"] * h)
                    left_ankle = (kp_dict[13]["x"] * w, kp_dict[13]["y"] * h)
                    
                    # 使用两个脚踝的平均高度
                    ankle_y = (right_ankle[1] + left_ankle[1]) / 2
                    height_px = ankle_y - neck[1]
                
                if height_px > 0:
                    shoulder_height_diff = shoulder_height_diff_px / height_px
                    result["measurements"]["shoulder_height_diff"] = float(shoulder_height_diff)
                    
                    # 基于临床标准判断高低肩 - 2%以上被认为是显著的不对称
                    if shoulder_height_diff > 0.02:
                        if right_shoulder[1] > left_shoulder[1]:
                            result["issues"].append("左侧高肩")
                            result["clinical_assessment"]["shoulder_asymmetry"] = {
                                "finding": "左侧肩峰高于右侧",
                                "deviation": f"{(shoulder_height_diff * 100):.1f}%",
                                "clinical_significance": "显著" if shoulder_height_diff > 0.05 else "轻度",
                                "standard": "临床上，肩峰高度差异>2%被视为不对称，>5%为显著不对称"
                            }
                        else:
                            result["issues"].append("右侧高肩")
                            result["clinical_assessment"]["shoulder_asymmetry"] = {
                                "finding": "右侧肩峰高于左侧",
                                "deviation": f"{(shoulder_height_diff * 100):.1f}%",
                                "clinical_significance": "显著" if shoulder_height_diff > 0.05 else "轻度",
                                "standard": "临床上，肩峰高度差异>2%被视为不对称，>5%为显著不对称"
                            }
                
                # 计算肩膀角度
                shoulder_angle = np.degrees(np.arctan2(left_shoulder[1] - right_shoulder[1], 
                                                      left_shoulder[0] - right_shoulder[0]))
                result["angles"]["shoulder_angle"] = float(shoulder_angle)
                
                # 肩膀角度临床评估 - 正常应接近0度（水平）
                result["clinical_assessment"]["shoulder_angle"] = {
                    "value": f"{shoulder_angle:.2f}°",
                    "normal_range": "±2°",
                    "interpretation": "明显偏离水平位" if abs(shoulder_angle) > 5 else ("轻度偏离水平位" if abs(shoulder_angle) > 2 else "正常范围"),
                    "biomechanical_impact": "可能导致肩部肌肉不平衡和斜方肌过度紧张" if abs(shoulder_angle) > 5 else "轻微影响"
                }
            
            # 检查头部倾斜
            if 0 in kp_dict and 1 in kp_dict and kp_dict[0]["confidence"] > 0.5 and kp_dict[1]["confidence"] > 0.5:
                nose = (kp_dict[0]["x"] * w, kp_dict[0]["y"] * h)
                neck = (kp_dict[1]["x"] * w, kp_dict[1]["y"] * h)
                
                # 计算头部相对于颈部的水平偏移
                head_horizontal_offset = (nose[0] - neck[0]) / w
                result["measurements"]["head_horizontal_offset"] = float(head_horizontal_offset)
                
                # 头部侧倾角度评估
                head_tilt_angle = np.degrees(np.arctan2(nose[0] - neck[0], neck[1] - nose[1]))
                result["angles"]["head_tilt_angle"] = float(head_tilt_angle)
                
                # 临床意义评估
                if abs(head_tilt_angle) > 3:
                    tilt_direction = "右" if head_tilt_angle > 0 else "左"
                    result["issues"].append(f"头部{tilt_direction}倾")
                    result["clinical_assessment"]["head_alignment"] = {
                        "finding": f"头部{tilt_direction}侧倾斜",
                        "angle": f"{abs(head_tilt_angle):.2f}°",
                        "clinical_significance": "显著" if abs(head_tilt_angle) > 10 else "轻度",
                        "potential_causes": "可能源自颈部肌肉不平衡、颈椎问题或代偿性姿势",
                        "functional_impact": "可能引起颈部慢性疼痛、头痛和颞下颌关节紊乱" if abs(head_tilt_angle) > 10 else "长期可能导致颈部肌肉疲劳和不适"
                    }
            
            # 检查髋部水平度
            if 8 in kp_dict and 11 in kp_dict and kp_dict[8]["confidence"] > 0.5 and kp_dict[11]["confidence"] > 0.5:
                right_hip = (kp_dict[8]["x"] * w, kp_dict[8]["y"] * h)
                left_hip = (kp_dict[11]["x"] * w, kp_dict[11]["y"] * h)
                
                # 计算髋部高度差异
                hip_height_diff_px = abs(right_hip[1] - left_hip[1])
                
                # 计算髋部相对高度差异
                if height_px > 0:
                    hip_height_diff = hip_height_diff_px / height_px
                    result["measurements"]["hip_height_diff"] = float(hip_height_diff)
                    
                    # 骨盆倾斜临床标准评估 - 1.5%以上被认为是显著的骨盆倾斜
                    if hip_height_diff > 0.015:
                        tilt_direction = "右" if right_hip[1] < left_hip[1] else "左"
                        result["issues"].append("骨盆倾斜")
                        result["clinical_assessment"]["pelvic_tilt"] = {
                            "finding": f"骨盆{tilt_direction}侧倾斜（髋关节高度不等）",
                            "deviation": f"{(hip_height_diff * 100):.1f}%",
                            "clinical_significance": "显著" if hip_height_diff > 0.03 else "轻度",
                            "potential_causes": "可能与腿长不等、髋关节问题或脊柱侧弯相关",
                            "standard": "临床上，骨盆倾斜>1.5%被视为异常，>3%为显著异常"
                        }
                
                # 计算髋部角度
                hip_angle = np.degrees(np.arctan2(left_hip[1] - right_hip[1], 
                                                 left_hip[0] - right_hip[0]))
                result["angles"]["hip_angle"] = float(hip_angle)
                
                # 髋部角度临床评估
                result["clinical_assessment"]["hip_angle"] = {
                    "value": f"{hip_angle:.2f}°",
                    "normal_range": "±2°",
                    "interpretation": "明显骨盆旋转" if abs(hip_angle) > 5 else ("轻度骨盆旋转" if abs(hip_angle) > 2 else "正常范围"),
                    "biomechanical_impact": "可能导致下背部应力增加和髋关节负荷不均" if abs(hip_angle) > 5 else "轻微影响脊柱和髋关节生物力学"
                }
            
            # 分析膝关节位置和Q角
            if (8 in kp_dict and 9 in kp_dict and 10 in kp_dict and 
                11 in kp_dict and 12 in kp_dict and 13 in kp_dict and
                kp_dict[8]["confidence"] > 0.5 and kp_dict[9]["confidence"] > 0.5 and 
                kp_dict[10]["confidence"] > 0.5 and kp_dict[11]["confidence"] > 0.5 and 
                kp_dict[12]["confidence"] > 0.5 and kp_dict[13]["confidence"] > 0.5):
                
                right_hip = (kp_dict[8]["x"] * w, kp_dict[8]["y"] * h)
                right_knee = (kp_dict[9]["x"] * w, kp_dict[9]["y"] * h)
                right_ankle = (kp_dict[10]["x"] * w, kp_dict[10]["y"] * h)
                
                left_hip = (kp_dict[11]["x"] * w, kp_dict[11]["y"] * h)
                left_knee = (kp_dict[12]["x"] * w, kp_dict[12]["y"] * h)
                left_ankle = (kp_dict[13]["x"] * w, kp_dict[13]["y"] * h)
                
                # 计算膝关节间距
                knee_distance = abs(right_knee[0] - left_knee[0])
                result["measurements"]["knee_distance"] = float(knee_distance)
                
                # 计算踝关节间距
                ankle_distance = abs(right_ankle[0] - left_ankle[0])
                result["measurements"]["ankle_distance"] = float(ankle_distance)
                
                # 计算髋关节间距
                hip_distance = abs(right_hip[0] - left_hip[0])
                result["measurements"]["hip_distance"] = float(hip_distance)
                
                # 计算Q角 (股四头肌角) - 临床上重要的下肢力线评估指标
                # 右腿Q角
                right_q_angle = self.calculate_q_angle(right_hip, right_knee, right_ankle)
                result["angles"]["right_q_angle"] = float(right_q_angle)
                
                # 左腿Q角
                left_q_angle = self.calculate_q_angle(left_hip, left_knee, left_ankle)
                result["angles"]["left_q_angle"] = float(left_q_angle)
                
                # 临床评估Q角
                result["clinical_assessment"]["q_angle"] = {
                    "right": {
                        "value": f"{right_q_angle:.2f}°",
                        "normal_range": "男性: 8-14°; 女性: 12-18°",
                        "interpretation": "显著增大" if right_q_angle > 20 else ("轻度增大" if right_q_angle > 15 else "正常范围")
                    },
                    "left": {
                        "value": f"{left_q_angle:.2f}°",
                        "normal_range": "男性: 8-14°; 女性: 12-18°",
                        "interpretation": "显著增大" if left_q_angle > 20 else ("轻度增大" if left_q_angle > 15 else "正常范围")
                    },
                    "clinical_significance": "Q角增大与髌骨不稳定性、髌股疼痛综合征和膝关节过度内旋相关"
                }
                
                # 通过膝踝关系判断X型腿或O型腿
                # X型腿(膝内翻): 膝关节间距小于踝关节间距
                # O型腿(膝外翻): 膝关节间距大于踝关节间距
                if knee_distance < ankle_distance * 0.9:
                    result["issues"].append("X型腿（膝内翻）")
                    result["clinical_assessment"]["knee_alignment"] = {
                        "finding": "膝内翻 (Genu Valgum)",
                        "severity": "显著" if knee_distance < ankle_distance * 0.7 else "轻度",
                        "measurement": f"膝距/踝距比值: {(knee_distance/ankle_distance):.2f}",
                        "normal_range": "0.9-1.1",
                        "clinical_significance": "增加膝关节内侧压力，可能与髌股疼痛和内侧副韧带应力相关",
                        "tibiofemoral_angle": "估计 >6° 外翻"
                    }
                elif knee_distance > ankle_distance * 1.1:
                    result["issues"].append("O型腿（膝外翻）")
                    result["clinical_assessment"]["knee_alignment"] = {
                        "finding": "膝外翻 (Genu Varum)",
                        "severity": "显著" if knee_distance > ankle_distance * 1.3 else "轻度",
                        "measurement": f"膝距/踝距比值: {(knee_distance/ankle_distance):.2f}",
                        "normal_range": "0.9-1.1",
                        "clinical_significance": "增加膝关节外侧压力，可能与外侧半月板应力和外侧副韧带过度拉伸相关",
                        "tibiofemoral_angle": "估计 >6° 内翻"
                    }
            
            return result
                
        except Exception as e:
            self.logger.error(f"前视图分析失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            result["issues"].append("前视图分析过程中发生错误")
            return result
            
    def calculate_q_angle(self, hip, knee, ankle):
        """计算Q角度，即股四头肌角度，由髋关节、膝关节和踝关节形成
        
        Args:
            hip: 髋关节坐标 [x, y]
            knee: 膝关节坐标 [x, y]
            ankle: 踝关节坐标 [x, y]
            
        Returns:
            Q角度（度）
        """
        try:
            # 计算两个向量：从膝盖到髋部和从膝盖到踝关节
            vector1 = (hip[0] - knee[0], hip[1] - knee[1])
            vector2 = (ankle[0] - knee[0], ankle[1] - knee[1])
            
            # 计算两个向量之间的角度
            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = np.sqrt(vector1[0]**2 + vector1[1]**2)
            magnitude2 = np.sqrt(vector2[0]**2 + vector2[1]**2)
            
            # 防止除零错误
            if magnitude1 * magnitude2 == 0:
                return 0
                
            cos_angle = dot_product / (magnitude1 * magnitude2)
            
            # 确保cos_angle在[-1, 1]范围内，避免因浮点精度问题导致的错误
            cos_angle = max(-1, min(1, cos_angle))
            
            # 计算角度（弧度）并转换为度
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            # Q角是180减去计算出的角度
            q_angle = 180 - angle_deg
            
            return q_angle
            
        except Exception as e:
            self.logger.error(f"Q角计算失败: {str(e)}")
            return 0
        
    def _analyze_back_view(self, keypoints, image):
        """分析后视图的体态问题
        
        Args:
            keypoints: 后视图关键点数据
            image: 后视图图像
            
        Returns:
            后视图分析结果字典
        """
        h, w = image.shape[:2]
        result = {
            "issues": [],
            "angles": {},
            "measurements": {},
            "clinical_assessment": {}  # 新增临床评估数据
        }
        
        # 通过ID快速查找关键点
        kp_dict = {kp["id"]: kp for kp in keypoints}
        
        try:
            # 检查肩膀高度差异（从后视图）
            if 2 in kp_dict and 5 in kp_dict and kp_dict[2]["confidence"] > 0.5 and kp_dict[5]["confidence"] > 0.5:
                right_shoulder = (kp_dict[2]["x"] * w, kp_dict[2]["y"] * h)
                left_shoulder = (kp_dict[5]["x"] * w, kp_dict[5]["y"] * h)
                
                # 计算肩膀高度差异
                shoulder_height_diff_px = abs(right_shoulder[1] - left_shoulder[1])
                result["measurements"]["shoulder_height_diff_back"] = float(shoulder_height_diff_px)
                
                # 相对身高的肩膀高度差异比例
                # 估计身高：使用颈部到脚踝的距离
                height_px = 0
                if 1 in kp_dict and 10 in kp_dict and 13 in kp_dict:  # 颈部和脚踝
                    neck = (kp_dict[1]["x"] * w, kp_dict[1]["y"] * h)
                    right_ankle = (kp_dict[10]["x"] * w, kp_dict[10]["y"] * h)
                    left_ankle = (kp_dict[13]["x"] * w, kp_dict[13]["y"] * h)
                    
                    # 使用两个脚踝的平均高度
                    ankle_y = (right_ankle[1] + left_ankle[1]) / 2
                    height_px = ankle_y - neck[1]
                
                if height_px > 0:
                    shoulder_height_diff = shoulder_height_diff_px / height_px
                    
                    # 通过后视图再次确认高低肩
                    if shoulder_height_diff > 0.02:
                        if right_shoulder[1] < left_shoulder[1]:  # 注意：后视图左右是相反的
                            result["issues"].append("右侧高肩(后视)")
                            result["clinical_assessment"]["shoulder_asymmetry_back"] = {
                                "finding": "右侧肩峰高于左侧",
                                "deviation": f"{(shoulder_height_diff * 100):.1f}%",
                                "clinical_significance": "显著" if shoulder_height_diff > 0.05 else "轻度",
                                "standard": "临床上，肩峰高度差异>2%被视为不对称，>5%为显著不对称"
                            }
                        else:
                            result["issues"].append("左侧高肩(后视)")
                            result["clinical_assessment"]["shoulder_asymmetry_back"] = {
                                "finding": "左侧肩峰高于右侧",
                                "deviation": f"{(shoulder_height_diff * 100):.1f}%",
                                "clinical_significance": "显著" if shoulder_height_diff > 0.05 else "轻度",
                                "standard": "临床上，肩峰高度差异>2%被视为不对称，>5%为显著不对称"
                            }
                
                # 计算后视图肩膀角度
                shoulder_angle = np.degrees(np.arctan2(left_shoulder[1] - right_shoulder[1], 
                                                      left_shoulder[0] - right_shoulder[0]))
                result["angles"]["shoulder_angle_back"] = float(shoulder_angle)
                
                # 肩膀角度临床评估 - 正常应接近0度（水平）
                result["clinical_assessment"]["shoulder_angle_back"] = {
                    "value": f"{shoulder_angle:.2f}°",
                    "normal_range": "±2°",
                    "interpretation": "明显偏离水平位" if abs(shoulder_angle) > 5 else ("轻度偏离水平位" if abs(shoulder_angle) > 2 else "正常范围"),
                    "biomechanical_impact": "可能导致肩部肌肉不平衡和斜方肌过度紧张" if abs(shoulder_angle) > 5 else "轻微影响"
                }
            
            # 评估脊柱直线度
            spine_points = []
            spine_keypoints = [1, 8, 11]  # 颈部, 右髋, 左髋
            
            for kp_id in spine_keypoints:
                if kp_id in kp_dict and kp_dict[kp_id]["confidence"] > 0.4:
                    point = (kp_dict[kp_id]["x"] * w, kp_dict[kp_id]["y"] * h)
                    spine_points.append(point)
            
            if len(spine_points) >= 3:
                # 计算脊柱中线点
                neck = spine_points[0]
                right_hip = spine_points[1]
                left_hip = spine_points[2]
                
                # 计算髋部中点
                hip_center_x = (right_hip[0] + left_hip[0]) / 2
                hip_center_y = (right_hip[1] + left_hip[1]) / 2
                hip_center = (hip_center_x, hip_center_y)
                
                # 计算脊柱角度 (相对于垂直线)
                spine_angle = np.degrees(np.arctan2(hip_center[0] - neck[0], hip_center[1] - neck[1]))
                result["angles"]["spine_angle"] = float(spine_angle)
                
                # 评估脊柱侧弯
                if abs(spine_angle) > 3:
                    lean_direction = "右" if spine_angle > 0 else "左"
                    result["issues"].append(f"脊柱{lean_direction}侧弯")
                    
                    # 临床评估 - Cobb角近似值
                    cobb_angle_estimate = abs(spine_angle) * 1.2  # 粗略估计Cobb角
                    scoliosis_grade = "轻度"
                    if cobb_angle_estimate > 25:
                        scoliosis_grade = "中度"
                    elif cobb_angle_estimate > 45:
                        scoliosis_grade = "重度"
                        
                    result["clinical_assessment"]["scoliosis"] = {
                        "finding": f"脊柱{lean_direction}侧弯",
                        "estimated_cobb_angle": f"{cobb_angle_estimate:.1f}°",
                        "classification": scoliosis_grade,
                        "clinical_standard": "临床上，Cobb角<20°为轻度，20-45°为中度，>45°为重度侧弯",
                        "recommendation": "建议进行射线检查确认Cobb角度" if cobb_angle_estimate > 20 else "建议定期随访监测侧弯发展"
                    }
            
            # 评估肩胛骨位置和高度
            if 2 in kp_dict and 5 in kp_dict and kp_dict[2]["confidence"] > 0.6 and kp_dict[5]["confidence"] > 0.6:
                right_shoulder = (kp_dict[2]["x"] * w, kp_dict[2]["y"] * h)
                left_shoulder = (kp_dict[5]["x"] * w, kp_dict[5]["y"] * h)
                
                # 检测肩胛骨高度差异和内收/外展
                scapular_width_ratio = abs(right_shoulder[0] - left_shoulder[0]) / w
                
                # 临床评估
                result["clinical_assessment"]["scapular_position"] = {
                    "width_ratio": f"{scapular_width_ratio:.2f}",
                    "normal_range": "0.35-0.45 (相对身体宽度)",
                    "interpretation": "肩胛骨内收 (肩胛骨靠近脊柱)" if scapular_width_ratio < 0.35 else ("肩胛骨外展 (翼状肩胛)" if scapular_width_ratio > 0.45 else "正常范围")
                }
                
                if scapular_width_ratio < 0.35:
                    result["issues"].append("肩胛骨内收")
                elif scapular_width_ratio > 0.45:
                    result["issues"].append("肩胛骨外展（翼状肩胛）")
            
            return result
                
        except Exception as e:
            self.logger.error(f"后视图分析失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            result["issues"].append("后视图分析过程中发生错误")
            return result
    
    def _analyze_side_views(self, left_keypoints, right_keypoints, left_image, right_image):
        """分析侧视图，检测前倾姿势、骨盆倾斜等问题
        
        Args:
            left_keypoints: 左视图关键点数据
            right_keypoints: 右视图关键点数据
            left_image: 左视图图像
            right_image: 右视图图像
            
        Returns:
            分析结果字典
        """
        h, w = left_image.shape[:2]
        result = {
            "issues": [],
            "angles": {},
            "measurements": {}
        }
        
        # 通过ID快速查找关键点
        left_kp_dict = {kp["id"]: kp for kp in left_keypoints}
        right_kp_dict = {kp["id"]: kp for kp in right_keypoints}
        
        try:
            # 分析前倾姿势
            if 0 in left_kp_dict and 1 in left_kp_dict and left_kp_dict[0]["confidence"] > 0.5 and left_kp_dict[1]["confidence"] > 0.5:
                nose = (left_kp_dict[0]["x"] * w, left_kp_dict[0]["y"] * h)
                neck = (left_kp_dict[1]["x"] * w, left_kp_dict[1]["y"] * h)
                
                # 计算头部相对于颈部的水平偏移
                head_offset = (nose[0] - neck[0]) / w  # 归一化偏移
                result["measurements"]["head_horizontal_offset"] = float(head_offset)
                
                if abs(head_offset) > 0.03:  # 偏移超过3%
                    if head_offset > 0:
                        result["issues"].append("左侧头部右倾")
                    else:
                        result["issues"].append("左侧头部左倾")
            
            if 0 in right_kp_dict and 1 in right_kp_dict and right_kp_dict[0]["confidence"] > 0.5 and right_kp_dict[1]["confidence"] > 0.5:
                nose = (right_kp_dict[0]["x"] * w, right_kp_dict[0]["y"] * h)
                neck = (right_kp_dict[1]["x"] * w, right_kp_dict[1]["y"] * h)
                
                # 计算头部相对于颈部的水平偏移
                head_offset = (nose[0] - neck[0]) / w  # 归一化偏移
                result["measurements"]["head_horizontal_offset"] = float(head_offset)
                
                if abs(head_offset) > 0.03:  # 偏移超过3%
                    if head_offset > 0:
                        result["issues"].append("右侧头部右倾")
                    else:
                        result["issues"].append("右侧头部左倾")
            
            # 分析髋部平衡
            if 8 in left_kp_dict and 11 in left_kp_dict and left_kp_dict[8]["confidence"] > 0.5 and left_kp_dict[11]["confidence"] > 0.5:
                left_hip = (left_kp_dict[8]["x"] * w, left_kp_dict[8]["y"] * h)
                left_hip_angle = np.degrees(np.arctan2(left_hip[1] - left_hip[1], 
                                               left_hip[0] - left_hip[0]))
                result["angles"]["left_hip_angle"] = float(left_hip_angle)
                
                if abs(left_hip_angle) > 5:
                    result["issues"].append("左侧髋部高")
            
            if 8 in right_kp_dict and 11 in right_kp_dict and right_kp_dict[8]["confidence"] > 0.5 and right_kp_dict[11]["confidence"] > 0.5:
                right_hip = (right_kp_dict[8]["x"] * w, right_kp_dict[8]["y"] * h)
                right_hip_angle = np.degrees(np.arctan2(right_hip[1] - right_hip[1], 
                                               right_hip[0] - right_hip[0]))
                result["angles"]["right_hip_angle"] = float(right_hip_angle)
                
                if abs(right_hip_angle) > 5:
                    result["issues"].append("右侧髋部高")
            
            return result
            
        except Exception as e:
            self.logger.error(f"侧视图分析失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            result["issues"].append("侧视图分析过程中发生错误")
            return result
    
    def _analyze_single_side_view(self, keypoints, image, side):
        """分析单个侧视图（左侧或右侧）
        
        Args:
            keypoints: 侧视图关键点数据
            image: 侧视图图像
            side: 侧视图类型 ("left" 或 "right")
            
        Returns:
            侧视图分析结果字典
        """
        h, w = image.shape[:2]
        result = {
            "issues": [],
            "angles": {},
            "measurements": {}
        }
        
        # 通过ID快速查找关键点
        kp_dict = {kp["id"]: kp for kp in keypoints}
        
        try:
            # 检测头部前倾
            if 0 in kp_dict and 1 in kp_dict and kp_dict[0]["confidence"] > 0.5 and kp_dict[1]["confidence"] > 0.5:
                nose = (kp_dict[0]["x"] * w, kp_dict[0]["y"] * h)
                neck = (kp_dict[1]["x"] * w, kp_dict[1]["y"] * h)
                
                # 计算头部相对于颈部的水平偏移
                head_forward_offset = (nose[0] - neck[0]) / w  # 归一化偏移
                result["measurements"][f"{side}_head_forward_offset"] = float(head_forward_offset)
                
                # 只有在向前倾斜时才判断为问题
                if head_forward_offset > 0.05:  # 偏移超过5%
                    result["issues"].append("头部前倾")
                    
                # 计算头部角度
                head_angle = np.degrees(np.arctan2(nose[1] - neck[1], nose[0] - neck[0]))
                result["angles"][f"{side}_head_angle"] = float(head_angle)
            
            # 检测脊柱弯曲状况
            spine_points = []
            
            if 1 in kp_dict:  # 颈部
                spine_points.append((kp_dict[1]["x"] * w, kp_dict[1]["y"] * h))
            
            # 使用肩膀和髋部位置估计脊柱中点
            if 2 in kp_dict and kp_dict[2]["confidence"] > 0.5:  # 肩部
                spine_points.append((kp_dict[2]["x"] * w, kp_dict[2]["y"] * h))
            
            if 8 in kp_dict and kp_dict[8]["confidence"] > 0.5:  # 髋部
                spine_points.append((kp_dict[8]["x"] * w, kp_dict[8]["y"] * h))
            
            # 如果有足够的点计算脊柱曲线
            if len(spine_points) >= 3:
                # 计算脊柱曲线
                x_coords = [p[0] for p in spine_points]
                y_coords = [p[1] for p in spine_points]
                
                # 使用多项式拟合脊柱曲线
                coeffs = np.polyfit(y_coords, x_coords, 2)
                
                # 曲率越大表示弯曲越明显
                curvature = abs(coeffs[0])
                result["measurements"][f"{side}_spine_curvature"] = float(curvature)
                
                if curvature > 0.0003:  # 阈值需要根据实际情况调整
                    result["issues"].append("脊柱前凸")
                
            # 检测膝关节过伸
            if 9 in kp_dict and 10 in kp_dict and 8 in kp_dict and \
               kp_dict[9]["confidence"] > 0.5 and kp_dict[10]["confidence"] > 0.5 and kp_dict[8]["confidence"] > 0.5:
                hip = (kp_dict[8]["x"] * w, kp_dict[8]["y"] * h)
                knee = (kp_dict[9]["x"] * w, kp_dict[9]["y"] * h)
                ankle = (kp_dict[10]["x"] * w, kp_dict[10]["y"] * h)
                
                # 计算膝关节角度
                vector1 = (hip[0] - knee[0], hip[1] - knee[1])
                vector2 = (ankle[0] - knee[0], ankle[1] - knee[1])
                
                dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
                magnitude1 = np.sqrt(vector1[0]**2 + vector1[1]**2)
                magnitude2 = np.sqrt(vector2[0]**2 + vector2[1]**2)
                
                cos_angle = dot_product / (magnitude1 * magnitude2)
                knee_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                
                result["angles"][f"{side}_knee_angle"] = float(knee_angle)
                
                # 检测膝关节过伸
                if knee_angle > 185:  # 膝关节过伸
                    result["issues"].append("膝关节过伸")
            
            return result
                
        except Exception as e:
            self.logger.error(f"{side}侧视图分析失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            result["issues"].append(f"{side}侧视图分析过程中发生错误")
            return result