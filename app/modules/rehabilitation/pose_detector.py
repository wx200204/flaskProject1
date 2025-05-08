import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import json

class PoseDetector:
    """使用MediaPipe的姿势检测器"""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """初始化MediaPipe姿势检测器"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化姿势检测器
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0, 1 或 2，越高越精确但更慢
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 姿势评分计算器
        self.posture_calculator = PostureCalculator()
        
        # 当前帧和处理状态
        self.current_frame = None
        self.processed_frame = None
        self.keypoints = None
        self.score = 0
        self.processing = False
        self.feedback = []
        
        # 性能指标
        self.last_process_time = 0
        self.last_frame_time = 0
        self.fps = 0
        self.process_time = 0
        
        # 上次成功处理时间
        self.last_successful_detection = 0
        
        print("姿势检测器初始化完成")
    
    def process_frame(self, frame):
        """处理单帧图像并返回姿势信息"""
        if frame is None:
            print("警告: 传入空帧进行处理")
            return None
        
        # 检查帧是否有效
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            print("警告: 传入无效帧进行处理")
            return None
            
        try:
            # 记录开始时间
            start_time = time.time()
            self.processing = True
            
            # 使用MediaPipe进行姿势检测
            # 转换颜色空间 BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 处理图像
            results = self.pose.process(rgb_frame)
            
            # 检查结果
            if results is None:
                print("MediaPipe返回空结果")
                # 创建带有提示的帧
                height, width, _ = frame.shape
                cv2.putText(frame, "处理失败", (int(width*0.1), int(height*0.5)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                self.processed_frame = frame
                return None
            
            # 如果没有检测到姿势
            if not results.pose_landmarks:
                self.keypoints = None
                self.score = 0
                self.feedback = ["未检测到人体，请确保全身在画面中"]
                
                # 创建带有提示的帧
                height, width, _ = frame.shape
                cv2.putText(frame, "未检测到人体姿势", (int(width*0.1), int(height*0.5)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                self.processed_frame = frame
                return None
            
            # 提取关键点数据
            landmarks = results.pose_landmarks.landmark
            keypoints = self._landmarks_to_keypoints(landmarks, frame.shape)
            self.keypoints = keypoints
            self.last_successful_detection = time.time()
            
            # 分析姿势
            score, feedback = self.posture_calculator.analyze_posture(keypoints)
            self.score = score
            self.feedback = feedback
            
            # 绘制姿势关键点和连接线
            annotated_frame = frame.copy()
            self._draw_pose_landmarks(annotated_frame, results.pose_landmarks)
            
            # 添加评分和反馈
            height, width, _ = annotated_frame.shape
            
            # 计算处理时间
            end_time = time.time()
            self.process_time = end_time - start_time
            
            # 存储处理后的帧
            self.processed_frame = annotated_frame
            
            # 返回结果
            return {
                'frame': annotated_frame,
                'keypoints': keypoints,
                'score': score,
                'feedback': feedback
            }
        except Exception as e:
            print(f"姿势检测错误: {e}")
            # 创建错误提示帧
            try:
                height, width, _ = frame.shape
                cv2.putText(frame, f"处理错误: {str(e)[:30]}", (10, int(height*0.5)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.processed_frame = frame
            except:
                pass
            return None
        finally:
            self.processing = False
    
    def get_keypoints_json(self):
        """返回适合JSON传输的关键点数据"""
        if self.keypoints is None:
            return json.dumps({
                'detected': False,
                'message': '未检测到人体',
                'timestamp': time.time()
            })
        
        try:
            # 添加超时检查 - 如果上次成功检测时间超过3秒，返回未检测状态
            if time.time() - self.last_successful_detection > 3.0:
                print("警告: 关键点数据已过期")
                return json.dumps({
                    'detected': False,
                    'message': '关键点数据已过期，请重新站到画面中',
                    'timestamp': time.time()
                })
            
            # 转换关键点格式，确保坐标适合前端绘制
            simplified_keypoints = {}
            for name, point in self.keypoints.items():
                # 确保点数据结构完整
                if not isinstance(point, dict) or 'visibility' not in point:
                    continue
                    
                # 为前端使用保留数据
                simplified_keypoints[name] = {
                    'x': float(point['x']),
                    'y': float(point['y']),
                    'z': float(point.get('z', 0.0)),
                    'visibility': float(point['visibility']),
                    'score': float(point.get('score', point['visibility']))
                }
            
            # 确保返回非空数据
            if not simplified_keypoints:
                print("警告: 转换后的关键点数据为空")
                return json.dumps({
                    'detected': False,
                    'message': '关键点数据结构不完整',
                    'timestamp': time.time()
                })
            
            # 返回完整结构
            result = {
                'detected': True,
                'keypoints': simplified_keypoints,
                'score': self.score,
                'feedback': self.feedback,
                'timestamp': time.time()
            }
            
            # 记录关键点计数
            print(f"返回 {len(simplified_keypoints)} 个关键点")
            return json.dumps(result)
            
        except Exception as e:
            print(f"关键点JSON序列化错误: {str(e)}")
            return json.dumps({
                'detected': False,
                'message': f'JSON序列化错误: {str(e)}',
                'timestamp': time.time()
            })
    
    def _landmarks_to_keypoints(self, landmarks, frame_shape):
        """将MediaPipe姿势关键点转换为字典格式"""
        try:
            height, width, _ = frame_shape
            keypoints = {}
            
            # MediaPipe姿势关键点名称
            landmark_names = [
                'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
                'right_eye_inner', 'right_eye', 'right_eye_outer',
                'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
                'left_index', 'right_index', 'left_thumb', 'right_thumb',
                'left_hip', 'right_hip', 'left_knee', 'right_knee',
                'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
                'left_foot_index', 'right_foot_index'
            ]
            
            for i, landmark in enumerate(landmarks):
                if i < len(landmark_names):
                    name = landmark_names[i]
                    # 检查landmark对象是否有效
                    if hasattr(landmark, 'x') and hasattr(landmark, 'y') and hasattr(landmark, 'visibility'):
                        # 转换为像素坐标
                        x = max(0, min(width, landmark.x * width))
                        y = max(0, min(height, landmark.y * height))
                        z = landmark.z * width if hasattr(landmark, 'z') else 0.0
                        
                        # 改进可见性值 - 确保数值在0-1之间
                        visibility = max(0.0, min(1.0, landmark.visibility))
                        
                        keypoints[name] = {
                            'x': x,
                            'y': y,
                            'z': z,
                            'visibility': visibility,
                            'score': visibility  # 使用visibility作为score
                        }
            
            # 只在有足够关键点时打印
            if len(keypoints) > 15:
                print(f"检测到 {len(keypoints)} 个关键点")
            return keypoints
            
        except Exception as e:
            print(f"关键点转换错误: {e}")
            return {}
    
    def _draw_pose_landmarks(self, frame, landmarks):
        """在帧上绘制姿势关键点和连接线"""
        # 使用MediaPipe提供的绘图功能
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # 额外强调脊柱线段（颈部到臀部中心的连接）
        if landmarks:
            # 获取图像尺寸
            height, width, _ = frame.shape
            
            # 提取关键点
            lm = landmarks.landmark
            
            # 计算颈部中心点（左右肩膀的中点）
            if lm[11].visibility > 0.5 and lm[12].visibility > 0.5:
                neck_x = int((lm[11].x + lm[12].x) / 2 * width)
                neck_y = int((lm[11].y + lm[12].y) / 2 * height)
                
                # 计算臀部中心点（左右髋关节的中点）
                if lm[23].visibility > 0.5 and lm[24].visibility > 0.5:
                    hip_x = int((lm[23].x + lm[24].x) / 2 * width)
                    hip_y = int((lm[23].y + lm[24].y) / 2 * height)
                    
                    # 绘制脊柱线段（加粗）
                    cv2.line(frame, (neck_x, neck_y), (hip_x, hip_y), (0, 255, 0), 4)
        
        return frame


class PostureCalculator:
    """姿势评分和反馈计算器"""
    
    def analyze_posture(self, keypoints):
        """分析姿势并提供评分和反馈"""
        if not keypoints:
            return 0, ["未检测到姿势"]
        
        # 初始分数和反馈列表
        score = 100
        feedback = []
        
        # 检查关键点是否可见
        required_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'nose']
        for point_name in required_points:
            if point_name not in keypoints or keypoints[point_name]['visibility'] < 0.5:
                return 50, ["无法看到完整上半身，请确保肩部和臀部在画面中"]
        
        # 检查脊柱垂直度
        try:
            # 计算肩部中点
            mid_shoulder_x = (keypoints['left_shoulder']['x'] + keypoints['right_shoulder']['x']) / 2
            mid_shoulder_y = (keypoints['left_shoulder']['y'] + keypoints['right_shoulder']['y']) / 2
            
            # 计算臀部中点
            mid_hip_x = (keypoints['left_hip']['x'] + keypoints['right_hip']['x']) / 2
            mid_hip_y = (keypoints['left_hip']['y'] + keypoints['right_hip']['y']) / 2
            
            # 计算脊柱垂直偏差
            dx = abs(mid_shoulder_x - mid_hip_x)
            dy = abs(mid_shoulder_y - mid_hip_y) or 1  # 避免除零错误
            spine_angle_deviation = dx / dy
            
            if spine_angle_deviation > 0.15:
                score -= 40
                feedback.append("脊柱严重倾斜，请保持脊柱垂直")
            elif spine_angle_deviation > 0.08:
                score -= 20
                feedback.append("脊柱轻微倾斜，请稍微调整")
        except (KeyError, ZeroDivisionError) as e:
            # 如果关键点数据缺失
            pass
        
        # 检查肩膀平衡度
        try:
            shoulder_height_diff = abs(keypoints['left_shoulder']['y'] - keypoints['right_shoulder']['y'])
            shoulder_width = abs(keypoints['left_shoulder']['x'] - keypoints['right_shoulder']['x']) or 1
            shoulder_imbalance = shoulder_height_diff / shoulder_width
            
            if shoulder_imbalance > 0.15:
                score -= 30
                feedback.append("肩膀严重倾斜，请保持肩膀水平")
            elif shoulder_imbalance > 0.08:
                score -= 15
                feedback.append("肩膀轻微倾斜，请调整肩膀高度")
        except (KeyError, ZeroDivisionError) as e:
            # 如果关键点数据缺失
            pass
        
        # 检查头部位置（相对于肩部）
        try:
            # 如果能够检测到鼻子和耳朵
            if 'nose' in keypoints and 'left_ear' in keypoints and 'right_ear' in keypoints:
                # 计算耳朵中点
                mid_ear_x = (keypoints['left_ear']['x'] + keypoints['right_ear']['x']) / 2
                mid_ear_y = (keypoints['left_ear']['y'] + keypoints['right_ear']['y']) / 2
                
                # 计算鼻子和耳朵中点的水平差距
                nose_ear_dx = keypoints['nose']['x'] - mid_ear_x
                shoulder_width = abs(keypoints['left_shoulder']['x'] - keypoints['right_shoulder']['x']) or 1
                head_forward_tilt = nose_ear_dx / shoulder_width
                
                if abs(head_forward_tilt) > 0.3:
                    score -= 20
                    if head_forward_tilt > 0:
                        feedback.append("头部前倾，请保持头部直立")
                    else:
                        feedback.append("头部后仰，请保持头部直立")
        except (KeyError, ZeroDivisionError) as e:
            # 如果关键点数据缺失
            pass
        
        # 确保分数在0-100之间
        score = max(0, min(100, score))
        
        # 如果没有问题，添加正面反馈
        if not feedback:
            feedback.append("姿势良好，请保持")
        
        return round(score), feedback 