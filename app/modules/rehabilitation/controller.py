import time
import threading
import json
from flask import jsonify, current_app
import os
import cv2
from app.utils.decorators import api_error_handler

try:
    from .video_processor import VideoProcessor
    from .posture_analyzer import PostureAnalyzer
    from .voice_feedback import VoiceFeedback
except ImportError:
    # 兼容直接引入的情况
    from app.modules.rehabilitation.video_processor import VideoProcessor
    from app.modules.rehabilitation.posture_analyzer import PostureAnalyzer
    from app.modules.rehabilitation.voice_feedback import VoiceFeedback

class RehabilitationController:
    """康复指导控制器，协调视频处理、姿态分析和反馈系统"""
    
    def __init__(self):
        """初始化控制器"""
        self.running = False
        self.session_id = None
        self.start_time = None
        self.video_processor = None
        self.posture_analyzer = PostureAnalyzer()
        self.voice_feedback = VoiceFeedback()
        self.analysis_results = []
        self.last_analysis_time = 0
        self.processing_thread = None
        self.stop_flag = threading.Event()
        self.template = "标准直立姿势"  # 默认模板
        self.error_message = None
        self.video_initialized = False
        
        # 保存状态的锁
        self.state_lock = threading.Lock()
        
        # 延迟初始化视频处理器，只创建实例但不启动摄像头
        try:
            self._create_video_processor()
        except Exception as e:
            self.error_message = f"创建视频处理器失败: {str(e)}"
            print(f"警告: {self.error_message}")
            # 不抛出异常，允许在没有摄像头的情况下创建控制器
        
        self.feedback_interval = 3.0  # 语音反馈间隔秒数
        self.last_feedback_time = 0
        
    def _create_video_processor(self):
        """创建视频处理器实例但不初始化摄像头"""
        camera_id = 0  # 默认摄像头ID
        width = 640
        height = 480
        
        # 使用应用配置（如果可用）
        try:
            if current_app and current_app.config:
                camera_id = current_app.config.get('CAMERA_ID', camera_id)
                width = current_app.config.get('VIDEO_WIDTH', width)
                height = current_app.config.get('VIDEO_HEIGHT', height)
        except Exception as e:
            print(f"获取配置失败: {str(e)}")
        
        # 创建视频处理器
        self.video_processor = VideoProcessor(
            camera_id=camera_id,
            width=width,
            height=height
        )
        
        # 注册回调
        if hasattr(self, 'on_frame_processed'):
            self.video_processor.on_frame_processed = self.on_frame_processed
        
        print("视频处理器已创建，等待请求时才会初始化摄像头")
    
    def initialize_video_processor(self):
        """初始化视频处理器（如果需要）并启动摄像头"""
        # 如果视频处理器未创建，先创建
        if self.video_processor is None:
            self._create_video_processor()
        
        # 注册回调（如果还没有设置）
        if hasattr(self, 'on_frame_processed'):
            self.video_processor.on_frame_processed = self.on_frame_processed
        
        # 标记已初始化
        self.video_initialized = True
        
        print("视频处理器已完成初始化，摄像头将在需要时启动")
        
    def on_frame_processed(self, result):
        """帧处理完成的回调函数"""
        try:
            if result and isinstance(result, dict):
                # 保存分析结果
                with self.state_lock:
                    self.last_analysis_time = time.time()
                    # 只保留最新的结果
                    if len(self.analysis_results) > 10:
                        self.analysis_results.pop(0)
                    self.analysis_results.append(result)
                
                # 提供反馈
                if result.get('feedback'):
                    # 检查是否需要提供语音反馈
                    current_time = time.time()
                    if current_time - self.last_feedback_time > self.feedback_interval:
                        feedback_text = "; ".join(result['feedback'])
                        self.voice_feedback.speak(feedback_text)
                        self.last_feedback_time = current_time
        except Exception as e:
            print(f"处理帧回调错误: {str(e)}")
        
    def start_session(self):
        """开始康复指导会话"""
        if self.running:
            return {"status": "already_running", "message": "会话已经在运行"}
        
        with self.state_lock:
            # 确保视频处理器已初始化
            if not self.video_initialized:
                self.initialize_video_processor()
            
            # 确保视频处理器存在
            if not self.video_processor:
                return {
                    "status": "error", 
                    "message": "视频处理器未初始化，无法启动会话"
                }
            
            # 启动视频处理
            if not self.video_processor.running:
                start_success = self.video_processor.start()
                if not start_success:
                    return {
                        "status": "error", 
                        "message": "摄像头初始化失败，请检查设备连接和权限"
                    }
            
            # 记录会话信息
            self.session_id = f"session_{int(time.time())}"
            self.start_time = time.time()
            
            # 重置分析结果
            self.analysis_results = []
            
            # 启动分析线程
            self.running = True
            self.stop_flag.clear()
            
            self.processing_thread = threading.Thread(target=self._analysis_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            print(f"康复指导会话已启动: {self.session_id}")
            
            return {"status": "success", "session_id": self.session_id}
        
    def _analysis_loop(self):
        """姿态分析循环"""
        last_feedback_content = None  # 记录上一次的反馈内容
        consecutive_similar_feedbacks = 0  # 连续相似反馈的次数
        human_detected = False  # 是否检测到人体
        last_human_status_time = 0  # 上次人体状态变化的时间
        
        while self.running:
            # 获取一帧并分析
            frame = self.video_processor.get_frame()
            if frame is not None:
                # 获取姿势关键点 - 现在返回的是 (landmarks, annotated_image) 元组
                pose_result = self.video_processor.extract_pose_keypoints(frame)
                current_time = time.time()
                
                # 处理未检测到人体的情况
                if not pose_result or pose_result[0] is None:
                    if human_detected:  # 之前检测到人体，现在没有
                        human_detected = False
                        last_human_status_time = current_time
                        # 人体刚消失时提醒一次
                        self.voice_feedback.speak("未检测到人体姿势，请确保您完整地出现在画面中")
                    else:  # 持续未检测到人体
                        # 每10秒才提醒一次
                        if current_time - last_human_status_time > 10.0:
                            self.voice_feedback.speak("未检测到人体姿势，请确保您完整地出现在画面中")
                            last_human_status_time = current_time
                    continue
                
                # 成功检测到关键点
                landmarks, annotated_image = pose_result
                
                # 确保有有效的关键点
                if landmarks is not None:
                    # 检测到人体状态变化
                    if not human_detected:
                        human_detected = True
                        last_human_status_time = current_time
                        # 刚检测到人体时提醒一次
                        self.voice_feedback.speak("已检测到人体，开始分析姿势")
                    
                    # 转换为旧格式以兼容后续代码
                    pose_data = {
                        'landmarks': landmarks,
                        'annotated_frame': annotated_image
                    }
                    
                    # 分析姿态
                    angles = self.posture_analyzer.calculate_spine_angles(landmarks)
                    if angles:
                        result = self.posture_analyzer.compare_with_template(angles, self.template)
                        
                        if result:
                            # 添加到结果历史
                            self.analysis_results.append(result)
                            if len(self.analysis_results) > 100:  # 只保留最近100条结果
                                self.analysis_results.pop(0)
                                    
                            # 提供语音反馈（有间隔地）
                            current_time = time.time()
                            
                            # 检查是否需要提供语音反馈
                            if current_time - self.last_feedback_time >= self.feedback_interval:
                                # 判断当前反馈与上次的是否相似
                                current_feedback = "；".join(result["feedback"]) if result["feedback"] else ""
                                
                                # 如果分数足够高，不需要反馈或表扬
                                if result["score"] >= 90:
                                    # 只有在持续保持良好姿势超过10秒才给予表扬，且与上次不同
                                    if current_time - last_human_status_time > 10.0:
                                        feedback_text = "姿势非常好，请继续保持"
                                        # 只有在之前不是好评价时才播放这个反馈
                                        if last_feedback_content != feedback_text:
                                            self.voice_feedback.speak(feedback_text)
                                            last_feedback_content = feedback_text
                                            self.last_feedback_time = current_time
                                            consecutive_similar_feedbacks = 0
                                            last_human_status_time = current_time  # 重置计时器
                                elif result["score"] < 70:  # 分数较低时提供具体反馈
                                    # 确保反馈内容不为空
                                    if current_feedback:
                                        # 如果当前反馈与上次相似
                                        if current_feedback == last_feedback_content:
                                            consecutive_similar_feedbacks += 1
                                            # 连续3次相似反馈，增加反馈间隔以防重复
                                            if consecutive_similar_feedbacks >= 3:
                                                self.feedback_interval = min(15.0, self.feedback_interval + 2.0)
                                            # 即使相似，也要间隔足够长的时间才播放
                                            if consecutive_similar_feedbacks < 5 or current_time - self.last_feedback_time >= 20.0:
                                                self.voice_feedback.speak(current_feedback)
                                                self.last_feedback_time = current_time
                                        else:
                                            # 不相似的反馈，正常播放
                                            self.voice_feedback.speak(current_feedback)
                                            last_feedback_content = current_feedback
                                            self.last_feedback_time = current_time
                                            consecutive_similar_feedbacks = 0
                                            # 重置反馈间隔
                                            self.feedback_interval = 3.0
            
            # 适当的休眠以减少CPU使用
            time.sleep(0.05)
    
    def get_latest_result(self):
        """获取最新的分析结果"""
        if not self.analysis_results:
            return {"status": "no_data"}
            
        latest = self.analysis_results[-1]
        fps = self.video_processor.fps
        
        return {
            "status": "success",
            "data": latest,
            "fps": fps,
            "template": self.template
        }
        
    def change_template(self, template_name):
        """更改当前使用的姿势模板"""
        if template_name in self.posture_analyzer.posture_templates:
            self.template = template_name
            self.voice_feedback.speak(f"已切换到{template_name}模板", priority=True)
            return {"status": "success", "template": template_name}
        else:
            return {"status": "error", "message": f"找不到模板: {template_name}"}
            
    def get_available_templates(self):
        """获取所有可用的姿势模板"""
        return {
            "status": "success",
            "templates": list(self.posture_analyzer.posture_templates.keys())
        }
        
    def stop_session(self):
        """停止康复指导会话"""
        self.running = False
        
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.stop_flag.set()
            self.processing_thread.join(timeout=1.0)
            
        self.video_processor.stop()
        self.voice_feedback.stop()
        
        return {"status": "success", "message": "康复指导会话已停止"}

    def analyze_single_frame(self, frame, template_name=None):
        """分析单帧图像中的姿势
        
        Args:
            frame: 图像帧
            template_name: 比对的模板名称，如果为None则使用当前模板
            
        Returns:
            分析结果字典
        """
        if template_name is None:
            template_name = self.template
            
        if frame is None or frame.size == 0:
            return {
                "status": "error",
                "message": "无效的图像数据"
            }
            
        # 使用视频处理器检测姿势
        pose_data = self.video_processor.extract_pose_keypoints(frame)
        landmarks, annotated_image = pose_data if pose_data else (None, None)
        
        if landmarks is None:
            return {
                "status": "not_detected",
                "message": "未检测到人体姿势，请确保完整地出现在画面中"
            }
            
        # 分析姿势角度
        angles = self.posture_analyzer.calculate_spine_angles(landmarks)
        
        if not angles:
            return {
                "status": "error", 
                "message": "无法分析姿势角度，请调整姿势"
            }
            
        # 与模板比较
        result = self.posture_analyzer.compare_with_template(angles, template_name)
        
        if not result:
            return {
                "status": "error",
                "message": "姿势比对失败"
            }
            
        # 基于比对结果，添加更多信息
        template_description = ""
        template_instruction = ""
        if template_name in self.posture_analyzer.posture_templates:
            template = self.posture_analyzer.posture_templates[template_name]
            template_description = template.get("description", "")
            template_instruction = template.get("instruction", "")
            
        # 构建完整结果
        analysis_result = {
            "status": "success",
            "timestamp": time.time(),
            "score": result["score"],
            "status_text": result["status"],
            "feedback": result["feedback"],
            "angles": angles,
            "landmarks": landmarks,
            "annotated_image": annotated_image,
            "template": {
                "name": template_name,
                "description": template_description,
                "instruction": template_instruction
            }
        }
        
        # 如果有关键偏差，添加语音反馈
        if result["score"] < 70 and hasattr(self, "voice_feedback"):
            feedback_text = "；".join(result["feedback"])
            self.voice_feedback.speak(feedback_text)
            
        return analysis_result
        
    def get_progress_metrics(self, session_id=None):
        """获取康复训练的进度指标
        
        Args:
            session_id: 会话ID，如果为None则使用当前会话
            
        Returns:
            进度指标字典
        """
        if not self.analysis_results:
            return {
                "status": "no_data",
                "message": "暂无训练数据"
            }
            
        # 计算评分趋势
        scores = [result.get("score", 0) for result in self.analysis_results if "score" in result]
        
        if not scores:
            return {
                "status": "no_scores",
                "message": "暂无评分数据"
            }
            
        # 计算评分统计数据
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # 计算评分趋势
        trend = 0
        if len(scores) >= 5:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            trend = second_avg - first_avg
            
        # 计算稳定性 (评分的标准差)
        if len(scores) >= 3:
            mean = avg_score
            variance = sum((x - mean) ** 2 for x in scores) / len(scores)
            stability = 100 - min(100, (variance ** 0.5) * 5)  # 转换为0-100的稳定性指标
        else:
            stability = 0
            
        # 计算在不同状态下的时间比例
        status_counts = {"优秀": 0, "良好": 0, "一般": 0, "需要改进": 0}
        for result in self.analysis_results:
            if "status" in result:
                status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
                
        total = sum(status_counts.values())
        status_percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in status_counts.items()}
        
        return {
            "status": "success",
            "metrics": {
                "avg_score": round(avg_score, 1),
                "min_score": min_score,
                "max_score": max_score,
                "trend": round(trend, 2),  # 正值表示提升，负值表示下降
                "stability": round(stability, 1),  # 0-100，越高越稳定
                "status_distribution": status_percentages,
                "samples": len(scores)
            },
            "session_info": {
                "session_id": session_id or self.session_id,
                "duration": time.time() - self.start_time if self.start_time else 0,
                "template": self.template
            }
        }
    
    def get_rehab_templates(self, category=None):
        """获取康复训练模板列表
        
        Args:
            category: 模板类别，如果为None则返回所有模板
            
        Returns:
            模板列表
        """
        templates = []
        
        for name, template in self.posture_analyzer.posture_templates.items():
            # 提取模板分类（如果有）
            template_category = template.get("category", "general")
            
            # 如果指定了类别且不匹配，跳过
            if category and template_category != category:
                continue
                
            templates.append({
                "name": name,
                "description": template.get("description", ""),
                "instruction": template.get("instruction", ""),
                "category": template_category,
                "difficulty": template.get("difficulty", "medium")
            })
            
        return {
            "status": "success",
            "templates": templates
        }

    @property
    def is_running(self):
        """检查康复指导会话是否在运行"""
        return self.running

    @property
    def is_camera_ready(self):
        """检查摄像头是否就绪，不会初始化摄像头"""
        if not self.video_initialized or not self.video_processor:
            return False
        return getattr(self.video_processor, 'camera_initialized', False)

    @property
    def is_pose_detector_ready(self):
        """检查姿态检测器是否就绪，不会初始化摄像头"""
        if not self.video_initialized or not self.video_processor:
            return False
        return hasattr(self.video_processor, 'pose_detector') and self.video_processor.pose_detector is not None 