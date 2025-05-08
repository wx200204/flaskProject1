import cv2
import numpy as np
import threading
import queue
import time
from pathlib import Path
from threading import Thread
import os
from app.utils.decorators import api_error_handler
from app.models.rehab.pose_detector import PoseDetector

class VideoProcessor:
    """视频处理器，负责捕获和处理摄像头视频"""
    
    def __init__(self, camera_id=0, width=640, height=480):
        """初始化视频处理器"""
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.stop_flag = threading.Event()
        self.processing_thread = None
        self.frame_buffer = None
        self.fps = 0
        self.last_frame_time = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.camera_error = False
        self.error_message = None
        self.active_clients = 0  # 跟踪活跃客户端数量
        self.last_client_activity = time.time()
        self.camera_initialized = False  # 标记摄像头是否已初始化
        
        # 姿势检测器 - 使用我们自定义的版本
        self.pose_detector = PoseDetector()
        
        # 帧队列 - 最大容量为5帧，用于非阻塞获取
        self.frame_queue = queue.Queue(maxsize=5)
        
        # 回调函数
        self.on_frame_processed = None
        
        # 创建一个空白帧作为默认帧
        self._create_placeholder_frame("摄像头未初始化，请点击开始按钮")
        
        print("视频处理器已创建，等待用户启动摄像头")
    
    def _create_placeholder_frame(self, message, sub_message=None):
        """创建一个占位帧"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if sub_message:
            cv2.putText(frame, sub_message, (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        self.frame_buffer = frame
    
    def initialize_camera(self):
        """初始化摄像头"""
        self.camera_error = False
        self.error_message = None
        
        # 尝试多种摄像头选项
        camera_options = [
            # 常规方式
            (self.camera_id, {}),
            # 尝试不同API
            (self.camera_id, {'apiPreference': cv2.CAP_ANY}),
            (self.camera_id, {'apiPreference': cv2.CAP_DSHOW}),
            (self.camera_id, {'apiPreference': cv2.CAP_MSMF}),
            # 尝试更多API - Windows特有
            (self.camera_id, {'apiPreference': cv2.CAP_WINRT}),
            (self.camera_id, {'apiPreference': cv2.CAP_AVFOUNDATION}),  # macOS
            (self.camera_id, {'apiPreference': cv2.CAP_V4L2}),  # Linux
            # 尝试不同的摄像头ID
            (0, {}),
            (0, {'apiPreference': cv2.CAP_ANY}),
            (1, {}),
            (1, {'apiPreference': cv2.CAP_ANY}),
            # 外部设备可能在更高索引
            (2, {}),
            (3, {})
        ]
        
        # 添加特殊处理: 寻找合适的摄像头设备
        try:
            # 在LINUX和macOS系统尝试列出设备
            if os.name == 'posix':  # Linux, macOS
                import glob
                v4l_devices = glob.glob('/dev/video*')
                for i, device in enumerate(v4l_devices):
                    device_id = int(device.replace('/dev/video', ''))
                    if device_id not in [option[0] for option in camera_options]:
                        camera_options.append((device_id, {}))
                        camera_options.append((device_id, {'apiPreference': cv2.CAP_V4L2}))
                print(f"找到 {len(v4l_devices)} 个摄像头设备: {v4l_devices}")
        except Exception as e:
            print(f"查找摄像头设备时出错: {e}")
        
        # 初始化前先释放可能的之前的摄像头
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"释放旧摄像头资源出错: {e}")
            # 完全释放资源
            self.cap = None
            # 等待更长时间确保资源释放，特别是在Windows上
            time.sleep(1.0)  # 增加等待时间
        
        # 尝试每个选项
        success = False
        attempts = 0
        last_error = None
        first_frames = []  # 存储第一帧用于验证
        
        for camera_id, params in camera_options:
            attempts += 1
            try:
                print(f"尝试初始化摄像头 ID={camera_id}, 参数={params}")
                
                # 防止仍在运行的摄像头干扰
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                    time.sleep(0.5)
                
                # 创建VideoCapture对象
                if params:
                    try:
                        self.cap = cv2.VideoCapture(camera_id, **params)
                    except Exception as e:
                        print(f"创建VideoCapture失败: {e}")
                        continue
                else:
                    try:
                        self.cap = cv2.VideoCapture(camera_id)
                    except Exception as e:
                        print(f"创建VideoCapture失败: {e}")
                        continue
                
                # 检查是否成功打开
                if not self.cap.isOpened():
                    print(f"摄像头 ID={camera_id} 打开失败")
                    continue
                
                # 尝试设置分辨率
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                except Exception as e:
                    print(f"设置分辨率失败: {e}")
                
                # 尝试设置更多参数
                try:
                    # 设置帧率
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    # 更改自动对焦、自动曝光设置
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 启用自动对焦
                    # 设置缓冲区大小
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception as e:
                    print(f"设置摄像头参数失败 (非致命): {e}")
                
                # 读取多帧验证摄像头
                valid_frames = 0
                for _ in range(10):  # 尝试10帧
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        valid_frames += 1
                        # 保存第一个有效帧
                        if valid_frames == 1:
                            first_frames.append((camera_id, params, frame.copy()))
                        # 读取超过3个有效帧即认为成功
                        if valid_frames >= 3:
                            break
                    time.sleep(0.1)  # 等待下一帧
                
                # 如果成功读取至少3帧，认为摄像头可用
                if valid_frames >= 3:
                    success = True
                    self.frame_buffer = first_frames[-1][2].copy()  # 使用最后测试的成功帧
                    print(f"摄像头初始化成功: ID={camera_id}, 选项={params}, 成功读取{valid_frames}帧")
                    break
                
                # 如果此选项没有读取到足够帧，关闭并尝试下一个
                print(f"摄像头 ID={camera_id} 无法读取足够帧 (只读取了{valid_frames}帧)，尝试下一个选项")
                self.cap.release()
                self.cap = None
                
            except Exception as e:
                last_error = e
                print(f"摄像头初始化失败 (ID={camera_id}): {e}")
                if self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
        
        # 如果所有选项都失败但至少成功获取到一帧
        if not success and first_frames:
            print("所有尝试均未获得稳定摄像头，但找到一些有效帧，使用最佳选项")
            # 使用第一个获取过有效帧的选项
            best_option = first_frames[0]
            camera_id, params, frame = best_option
            
            try:
                # 重新初始化最佳选项
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                if params:
                    self.cap = cv2.VideoCapture(camera_id, **params)
                else:
                    self.cap = cv2.VideoCapture(camera_id)
                
                # 保存帧缓冲
                self.frame_buffer = frame.copy()
                
                # 标记为部分成功
                print(f"使用部分工作的摄像头选项: ID={camera_id}, 参数={params}")
                success = True
            except Exception as e:
                print(f"重新初始化最佳选项失败: {e}")
                success = False
        
        # 如果所有选项都失败
        if not success:
            self.camera_error = True
            error_detail = f"{last_error}" if last_error else "未知错误"
            self.error_message = f"无法初始化摄像头: {error_detail}"
            print(f"所有 {attempts} 次摄像头初始化尝试均失败: {error_detail}")
            
            # 创建一个错误帧作为帧缓冲
            error_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(error_frame, "摄像头初始化失败", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_frame, error_detail[:50], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(error_frame, "请检查摄像头连接", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            self.frame_buffer = error_frame
            return False
        
        # 清空帧队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        
        # 预填充帧队列
        if self.frame_buffer is not None:
            try:
                self.frame_queue.put(self.frame_buffer.copy())
            except:
                pass
        
        self.last_frame_time = time.time()
        return True
    
    def register_client(self):
        """注册一个新的客户端"""
        self.active_clients += 1
        self.last_client_activity = time.time()
        print(f"视频处理器: 注册新客户端, 当前活跃客户端: {self.active_clients}")
        
        # 如果有客户端但没有运行，则启动
        if self.active_clients > 0 and not self.running:
            self.start()
    
    def unregister_client(self):
        """注销一个客户端"""
        if self.active_clients > 0:
            self.active_clients -= 1
        self.last_client_activity = time.time()
        print(f"视频处理器: 客户端注销, 剩余活跃客户端: {self.active_clients}")
        
        # 如果没有活跃客户端了，可以暂停处理以节省资源
        if self.active_clients == 0 and self.running:
            # 还是保持运行，但减少处理频率
            print("视频处理器: 无活跃客户端，将减少处理频率")
    
    def start(self):
        """启动视频处理"""
        if self.running:
            return True  # 已经运行中，不需要重新启动
        
        # 如果摄像头未初始化，先初始化摄像头
        if not self.camera_initialized:
            print("摄像头尚未初始化，现在进行初始化...")
            if not self.initialize_camera():
                self._create_placeholder_frame("摄像头初始化失败", 
                                             "请检查摄像头连接和浏览器权限")
                return False
            
            self.camera_initialized = True
        # 如果摄像头已关闭或有错误，尝试重新初始化
        elif self.cap is None or self.camera_error:
            print("摄像头已关闭或有错误，尝试重新初始化...")
            if not self.initialize_camera():
                return False
        
        # 开始处理线程
        self.running = True
        self.stop_flag.clear()
        
        # 创建并启动处理线程
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("视频处理器已启动")
        return True
    
    def stop(self):
        """停止视频处理"""
        self.running = False
        self.stop_flag.set()
        
        # 等待处理线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("视频处理器已停止")
    
    def get_frame(self):
        """获取当前处理后的帧"""
        # 如果未初始化且未显示错误，返回占位帧
        if not self.camera_initialized and not self.camera_error:
            if self.frame_buffer is None:
                self._create_placeholder_frame("摄像头未初始化", "点击开始按钮启动摄像头")
            return self.frame_buffer
        
        # 优先返回已处理的帧
        if self.running and hasattr(self, 'pose_detector') and self.pose_detector is not None and self.pose_detector.processed_frame is not None:
            return self.pose_detector.processed_frame
        
        # 如果没有处理后的帧，尝试返回原始帧
        if hasattr(self, 'frame_queue') and not self.frame_queue.empty():
            try:
                raw_frame = self.frame_queue.get_nowait()
                return raw_frame
            except queue.Empty:
                pass
        
        # 如果有帧缓冲，返回缓冲帧
        if self.frame_buffer is not None:
            return self.frame_buffer.copy()
        
        # 最后，如果有相机错误，返回错误帧
        if self.camera_error:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, "摄像头错误", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, self.error_message or "未知错误", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return frame
        
        # 无帧可用，创建空帧
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(frame, "等待摄像头...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame
    
    def extract_pose_keypoints(self, frame=None):
        """从帧中提取姿势关键点
        
        Args:
            frame: 可选的输入帧，如果为None则使用当前帧
            
        Returns:
            landmarks: 提取的关键点
            annotated_frame: 标注了关键点的帧
        """
        if frame is None:
            frame = self.frame_buffer.copy() if self.frame_buffer is not None else None
        
        if frame is None:
            return None, None
            
        try:
            # 使用我们修改后的PoseDetector，它不依赖mediapipe
            return self.pose_detector.detect_pose(frame)
        except Exception as e:
            print(f"姿势提取错误: {e}")
            return None, frame
            
    def get_keypoints_json(self):
        """获取关键点的JSON表示
        
        Returns:
            json_str: 关键点的JSON字符串
        """
        return self.pose_detector.get_keypoints_json()
    
    def _process_loop(self):
        """视频处理循环"""
        if not self.cap or not self.cap.isOpened():
            print("无法启动处理循环: 摄像头未打开")
            self.camera_error = True
            self.error_message = "摄像头未打开"
            self.running = False
            return
        
        # 重置计数器和时间
        self.frame_count = 0
        self.start_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 30
        last_camera_check = time.time()
        camera_check_interval = 10.0  # 每10秒检查一次摄像头状态
        
        # 清空帧队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        print("视频处理循环已启动")
        
        # 主处理循环
        while self.running and not self.stop_flag.is_set():
            try:
                # 检查客户端活动 - 如果长时间无活动，降低处理频率以节省资源
                if time.time() - self.last_client_activity > 5.0 and self.active_clients == 0:
                    time.sleep(0.5)  # 降低帧率到约2fps
                    continue
                elif time.time() - self.last_client_activity > 30.0:
                    # 长时间无活动，暂停摄像头处理但保持循环运行
                    time.sleep(1.0)
                    continue
                
                # 定期检查摄像头是否需要重新初始化
                current_time = time.time()
                if current_time - last_camera_check > camera_check_interval:
                    last_camera_check = current_time
                    
                    # 如果摄像头已关闭，尝试重新初始化
                    if self.cap is None or not self.cap.isOpened():
                        print("检测到摄像头已关闭，尝试重新初始化")
                        self.initialize_camera()
                
                # 检查摄像头是否已打开
                if self.cap is None or not self.cap.isOpened():
                    consecutive_errors += 1
                    print(f"摄像头未打开, 错误 #{consecutive_errors}/{max_consecutive_errors}")
                    
                    # 如果连续多次错误，尝试重新初始化
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"连续 {consecutive_errors} 次错误，尝试重新初始化摄像头")
                        self.initialize_camera()
                        consecutive_errors = 0
                    
                    time.sleep(0.1)
                    continue
                
                # 读取一帧
                ret, frame = self.cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    consecutive_errors += 1
                    print(f"无法读取视频帧, 错误 #{consecutive_errors}/{max_consecutive_errors}")
                    
                    # 如果连续多次无法读取，尝试重新初始化
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"连续 {consecutive_errors} 次无法读取帧，尝试重新初始化摄像头")
                        self.initialize_camera()
                        consecutive_errors = 0
                    
                    # 如果有缓冲帧，使用缓冲帧
                    if self.frame_buffer is not None:
                        # 将缓冲帧加入队列，确保始终有帧可用
                        try:
                            if not self.frame_queue.full():
                                self.frame_queue.put_nowait(self.frame_buffer.copy())
                        except:
                            pass
                    
                    time.sleep(0.1)
                    continue
                
                # 有效帧处理
                if frame is not None:
                    # 重置错误计数
                    consecutive_errors = 0
                    
                    # 更新FPS计算
                    current_time = time.time()
                    self.frame_count += 1
                    elapsed = current_time - self.start_time
                    
                    if elapsed >= 1.0:  # 每秒更新一次FPS
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.start_time = current_time
                    
                    # 处理帧 - 保存到缓冲区
                    self.frame_buffer = frame.copy()
                    
                    # 将原始帧加入队列，以便在姿势检测失败时能够显示原始帧
                    try:
                        # 清空队列，只保留最新帧
                        while not self.frame_queue.empty():
                            self.frame_queue.get_nowait()
                        # 添加新帧
                        self.frame_queue.put_nowait(frame.copy())
                    except Exception as e:
                        print(f"帧队列操作错误: {e}")
                    
                    # 姿势检测 - 只有有活跃客户端或最后5秒内有活动时才处理
                    if self.active_clients > 0 or (time.time() - self.last_client_activity < 5.0):
                        try:
                            # 处理前先检查姿势检测器是否已初始化
                            if hasattr(self, 'pose_detector') and self.pose_detector is not None:
                                self.pose_detector.process_frame(frame)
                                
                                # 如果有回调函数，处理结果
                                if self.on_frame_processed is not None:
                                    try:
                                        pose_data = self.extract_pose_keypoints(frame)
                                        if pose_data:
                                            self.on_frame_processed(pose_data)
                                    except Exception as callback_error:
                                        print(f"回调处理错误: {callback_error}")
                            else:
                                print("姿势检测器未初始化")
                        except Exception as pose_error:
                            print(f"姿势检测错误: {pose_error}")
                else:
                    time.sleep(0.01)  # 避免空循环占用CPU
            
            except Exception as e:
                print(f"视频处理循环错误: {e}")
                time.sleep(0.1)  # 避免错误情况下的高CPU使用率
    
    def switch_camera(self, camera_id):
        """切换摄像头"""
        if camera_id == self.camera_id:
            return True  # 如果是相同的摄像头，不需要切换
        
        # 暂停当前处理
        was_running = self.running
        if was_running:
            self.stop()
        
        # 更新摄像头ID
        self.camera_id = camera_id
        
        # 尝试初始化新摄像头
        success = self.initialize_camera()
        
        # 如果之前在运行，且初始化成功，重新启动视频处理
        if was_running and success:
            self.start()
        
        return success 

    def get_debug_info(self):
        """获取调试信息"""
        return {
            "camera_id": self.camera_id,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "running_time": time.time() - self.start_time,
            "camera_initialized": self.camera_initialized,
            "running": self.running,
            "camera_error": self.camera_error,
            "error_message": self.error_message
        }
        
    def get_skeleton_connections(self):
        """获取骨架连接信息，用于前端绘制骨架线条
        
        返回值:
            list: 每个元素为[起点索引, 终点索引]，表示骨架的一条线段
        """
        # 标准人体骨架连接关系
        connections = [
            # 躯干
            [11, 12],  # 左肩到右肩
            [11, 23],  # 左肩到左髋
            [12, 24],  # 右肩到右髋
            [23, 24],  # 左髋到右髋
            
            # 左臂
            [11, 13],  # 左肩到左肘
            [13, 15],  # 左肘到左手腕
            
            # 右臂
            [12, 14],  # 右肩到右肘
            [14, 16],  # 右肘到右手腕
            
            # 左腿
            [23, 25],  # 左髋到左膝
            [25, 27],  # 左膝到左踝
            
            # 右腿
            [24, 26],  # 右髋到右膝
            [26, 28],  # 右膝到右踝
            
            # 脸部连接（可选）
            [0, 4],    # 脸部轮廓
            [0, 1],    
            [1, 2],
            [2, 3],
            [3, 7],
            [0, 5],
            [5, 6],
            [6, 8],
            [9, 10]    # 嘴巴
        ]
        
        return connections
    
    def get_camera_options(self):
        """获取可用的摄像头选项"""
        camera_options = []
        
        # 尝试查找多个摄像头
        for i in range(10):  # 最多尝试10个摄像头索引
            try:
                temp_cap = cv2.VideoCapture(i)
                if temp_cap is not None and temp_cap.isOpened():
                    ret, frame = temp_cap.read()
                    if ret and frame is not None and frame.size > 0:
                        camera_name = f"摄像头 {i}"
                        # 尝试获取摄像头属性
                        try:
                            width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            camera_name = f"摄像头 {i} ({width}x{height})"
                        except:
                            pass
                        
                        camera_options.append({
                            "id": i,
                            "name": camera_name
                        })
                    temp_cap.release()
            except Exception as e:
                print(f"检测摄像头 {i} 时出错: {e}")
        
        return camera_options 