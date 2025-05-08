from flask import Blueprint, Response, current_app, jsonify, request
import cv2
import numpy as np
import time
import base64
import threading
import json
# 删除不兼容的导入，改用flask公共API
# from flask import _app_ctx_stack, has_app_context
# 延迟导入，避免循环导入
# from app.modules.rehabilitation.controller import RehabilitationController

# 修改蓝图名称，避免冲突
video_bp = Blueprint('video_stream', __name__, url_prefix='/video')

# 全局变量存储控制器实例，避免每次访问current_app
_controller_instance = None
_controller_lock = threading.Lock()
_active_clients = {}  # 存储活跃的客户端连接

# 获取控制器实例
def get_controller():
    global _controller_instance
    
    # 如果已有实例，直接返回
    if _controller_instance:
        return _controller_instance
    
    # 检查是否在应用上下文中的更安全方法
    try:
        with _controller_lock:
            if not _controller_instance:
                _controller_instance = current_app.extensions.get('rehab_controller')
            return _controller_instance
    except RuntimeError:
        # 捕获"处于应用上下文外"的运行时错误
        print("警告: 在应用上下文外尝试访问控制器")
        return None

def generate_video_feed():
    """生成视频帧流"""
    # 设置环境变量告诉OpenCV不要使用GUI
    cv2.setNumThreads(1)  # 限制OpenCV线程数，防止资源争用
    
    # 生成唯一的客户端ID
    client_id = f"client_{time.time()}_{id(threading.current_thread())}"
    client_start_time = time.time()
    print(f"新的视频流连接: {client_id}")
    
    # 注册客户端
    _active_clients[client_id] = {
        'start_time': client_start_time,
        'last_activity': time.time(),
        'is_active': True
    }
    
    try:
        controller = get_controller()
        if not controller:
            # 如果控制器不存在，返回错误图像
            blank_image = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_image, "康复指导未初始化", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(blank_image, "请刷新页面或联系管理员", (50, 280), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            _, buffer = cv2.imencode('.jpg', blank_image)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
        
        # 确保视频处理器已启动
        if hasattr(controller, 'video_processor'):
            if not controller.video_processor.running:
                print("启动视频处理器")
                start_success = controller.video_processor.start()
                if not start_success:
                    print("视频处理器启动失败")
                    # 创建并返回错误图像
                    blank_image = np.zeros((480, 640, 3), np.uint8)
                    cv2.putText(blank_image, "摄像头启动失败", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2)
                    if hasattr(controller.video_processor, 'error_message') and controller.video_processor.error_message:
                        cv2.putText(blank_image, controller.video_processor.error_message[:40], (50, 280), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(blank_image, "请确认摄像头连接并授予浏览器访问权限", (50, 320), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    _, buffer = cv2.imencode('.jpg', blank_image)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    # 返回第二帧，确保浏览器正确显示
                    time.sleep(0.5)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    return
            # 等待视频处理器完成初始化
            time.sleep(0.5)
        else:
            # 如果没有视频处理器，返回错误图像
            blank_image = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_image, "视频处理器未初始化", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2)
            cv2.putText(blank_image, "请刷新页面或联系管理员", (50, 280), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            _, buffer = cv2.imencode('.jpg', blank_image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
        
        start_time = time.time()
        frame_count = 0
        last_status_time = time.time()
        camera_status = "正在初始化摄像头..."
        retry_count = 0
        max_retries = 10
        
        # 发送初始帧，确保浏览器立即显示内容
        init_frame = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(init_frame, "正在连接摄像头...", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(init_frame, "准备中，请稍候...", (50, 280), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        _, buffer = cv2.imencode('.jpg', init_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # 添加一个短暂延迟，确保客户端收到初始帧
        time.sleep(0.2)
        
        # 故障计数器
        connection_failures = 0
        max_failures = 5
        
        while _active_clients.get(client_id, {}).get('is_active', False):
            try:
                # 更新客户端活动时间
                _active_clients[client_id]['last_activity'] = time.time()
                
                # 获取视频处理器实例
                if not hasattr(controller, 'video_processor') or controller.video_processor is None:
                    raise Exception("视频处理器不存在")
                
                # 检查摄像头状态
                if hasattr(controller.video_processor, 'camera_error') and controller.video_processor.camera_error:
                    # 使用错误信息帧
                    if hasattr(controller.video_processor, 'get_frame'):
                        frame = controller.video_processor.get_frame()  # 这将返回错误信息帧
                    else:
                        # 创建自定义错误帧
                        frame = np.zeros((480, 640, 3), np.uint8)
                        cv2.putText(frame, "摄像头错误", (50, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        error_msg = controller.video_processor.error_message if hasattr(controller.video_processor, 'error_message') else "未知错误"
                        cv2.putText(frame, error_msg[:40], (50, 280), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                else:
                    # 获取处理后的帧
                    frame = controller.video_processor.get_frame()
                
                if frame is None:
                    # 如果没有帧，创建等待状态帧
                    current_time = time.time()
                    
                    # 每3秒更新一次状态消息，给用户更好的反馈
                    if current_time - last_status_time > 3:
                        retry_count += 1
                        last_status_time = current_time
                        
                        if retry_count > max_retries:
                            camera_status = "摄像头无法启动，请检查设备连接"
                        else:
                            camera_status = f"等待摄像头响应... ({retry_count}/{max_retries})"
                    
                    # 生成等待状态帧
                    blank_image = np.zeros((480, 640, 3), np.uint8)
                    cv2.putText(blank_image, camera_status, (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(blank_image, "如长时间无响应，请检查摄像头权限", (50, 280), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cv2.putText(blank_image, "或尝试重新启动应用", (50, 310), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    # 添加按钮框
                    cv2.rectangle(blank_image, (245, 340), (395, 380), (70, 130, 180), -1)
                    cv2.rectangle(blank_image, (245, 340), (395, 380), (255, 255, 255), 1)
                    cv2.putText(blank_image, "刷新页面", (270, 367), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    
                    frame = blank_image
                    
                    # 增加连接失败计数
                    connection_failures += 1
                    if connection_failures >= max_failures:
                        print(f"连续 {connection_failures} 次无法获取帧，尝试重新初始化摄像头")
                        try:
                            if controller.video_processor:
                                # 尝试重新初始化摄像头
                                controller.video_processor.initialize_camera()
                                connection_failures = 0  # 重置计数器
                        except Exception as reinit_error:
                            print(f"重新初始化摄像头失败: {reinit_error}")
                    
                else:
                    # 收到帧，重置故障计数
                    connection_failures = 0
                    retry_count = 0
                    
                    # 计算FPS
                    frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    # 每秒重置一次计数
                    if elapsed > 1.0:
                        start_time = current_time
                        frame_count = 0
                    
                    # 添加FPS显示
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                # 确保帧有效
                if frame is None or frame.size == 0:
                    frame = np.zeros((480, 640, 3), np.uint8)
                    cv2.putText(frame, "无效帧", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 压缩质量调整 - 降低质量以提高传输速度
                try:
                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # 85%质量
                    _, buffer = cv2.imencode('.jpg', frame, encode_params)
                    
                    # 返回帧数据
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                except Exception as encode_error:
                    print(f"帧编码错误: {encode_error}")
                    # 尝试生成简单的错误帧
                    simple_frame = np.zeros((240, 320, 3), np.uint8)
                    cv2.putText(simple_frame, "帧处理错误", (50, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                    try:
                        _, simple_buffer = cv2.imencode('.jpg', simple_frame)
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + simple_buffer.tobytes() + b'\r\n')
                    except:
                        # 如果还是失败，发送一个最简单的帧
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xdb\x00C\x01\t\t\t\x0c\x0b\x0c\x18\r\r\x182!\x1c!22222222222222222222222222222222222222222222222222\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a&\'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xfe\xfe(\xa2\x8a\x00\xff\xd9' + b'\r\n')
                
                # 控制帧率，避免过高的CPU使用率
                time.sleep(0.03)  # 约30fps
                
            except Exception as e:
                print(f"处理视频帧时出错: {e}")
                # 生成错误帧
                try:
                    error_image = np.zeros((480, 640, 3), np.uint8)
                    cv2.putText(error_image, "视频流错误", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    error_msg = str(e)
                    # 分行显示错误信息
                    y_pos = 280
                    while error_msg and y_pos < 400:
                        display_msg = error_msg[:50]
                        error_msg = error_msg[50:] if len(error_msg) > 50 else ""
                        cv2.putText(error_image, display_msg, (50, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_pos += 30
                    
                    _, buffer = cv2.imencode('.jpg', error_image)
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                except Exception as inner_error:
                    print(f"生成错误帧失败: {inner_error}")
                    # 发送一个简单的JPEG图像
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xdb\x00C\x01\t\t\t\x0c\x0b\x0c\x18\r\r\x182!\x1c!22222222222222222222222222222222222222222222222222\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a&\'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xfe\xfe(\xa2\x8a\x00\xff\xd9' + b'\r\n')
                
                time.sleep(1)  # 错误情况下延长休眠时间
                
    except GeneratorExit:
        # 客户端断开连接，清理资源
        print(f"视频流客户端断开连接: {client_id}, 连接持续时间: {time.time() - client_start_time:.1f}秒")
        cleanup_client(client_id)
    except Exception as e:
        print(f"视频流生成器发生错误: {str(e)}")
    finally:
        # 确保客户端被标记为非活跃
        cleanup_client(client_id)

def cleanup_client(client_id):
    """清理客户端资源"""
    if client_id in _active_clients:
        _active_clients[client_id]['is_active'] = False
        
        # 获取连接数量
        active_count = sum(1 for client in _active_clients.values() if client['is_active'])
        print(f"客户端 {client_id} 已清理，剩余活跃连接: {active_count}")
        
        # 如果没有活跃客户端，通知视频处理器
        if active_count == 0:
            try:
                controller = get_controller()
                if controller and controller.video_processor:
                    controller.video_processor.unregister_client()
                    print("没有活跃的客户端连接，已通知视频处理器")
            except Exception as e:
                print(f"通知视频处理器时发生错误: {str(e)}")

@video_bp.route('/feed')
def video_feed():
    """视频流路由"""
    # 注册一个新的视频流客户端
    controller = get_controller()
    if controller and hasattr(controller, 'video_processor'):
        controller.video_processor.register_client()
        
        # 记录客户端活动，但不强制初始化摄像头
        print("新的视频流客户端已连接")
    
    # 设置响应头，禁止缓存
    response = Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    # 添加CORS和内容类型控制头
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@video_bp.route('/keypoints')
def get_keypoints():
    """获取当前骨骼关键点数据"""
    try:
        # 检查控制器是否初始化
        controller = get_controller()
        if controller is None:
            print("错误: 控制器未初始化，无法获取关键点")
            response = jsonify({
                'detected': False,
                'message': '视频处理服务未初始化',
                'timestamp': time.time()
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        # 检查视频处理器是否初始化
        if controller.video_processor is None:
            print("错误: 视频处理器未初始化")
            response = jsonify({
                'detected': False,
                'message': '视频处理服务未准备好',
                'timestamp': time.time()
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        
        # 登记客户端活动
        controller.video_processor.last_client_activity = time.time()
            
        # 获取关键点JSON数据 - 从video_processor获取而不是pose_detector
        keypoints_json = controller.video_processor.get_keypoints_json()
        if not keypoints_json:
            print("警告: 关键点数据为空")
            response = jsonify({
                'detected': False,
                'message': '未检测到姿势数据',
                'timestamp': time.time()
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        # 验证JSON数据格式是否正确
        try:
            keypoints_data = json.loads(keypoints_json)
            if 'detected' not in keypoints_data:
                print("警告: 关键点JSON数据格式不正确")
                keypoints_data = {
                    'detected': False,
                    'message': '数据格式错误',
                    'timestamp': time.time()
                }
        except Exception as e:
            print(f"JSON解析错误: {str(e)}")
            keypoints_data = {
                'detected': False,
                'message': f'数据解析错误: {str(e)}',
                'timestamp': time.time()
            }
        
        # 返回带CORS头的响应
        response = jsonify(keypoints_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        print(f"获取关键点数据时发生错误: {str(e)}")
        response = jsonify({
            'detected': False,
            'message': f'服务器错误: {str(e)}',
            'timestamp': time.time()
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@video_bp.route('/status')
def video_status():
    """获取视频流状态，不会初始化摄像头"""
    from app.modules.rehabilitation.controller import RehabilitationController
    
    try:
        controller = get_controller()
        
        # 获取状态，但不会初始化摄像头
        is_initialized = controller is not None
        is_running = is_initialized and controller.is_running
        camera_ready = is_initialized and controller.is_camera_ready
        pose_detector_ready = is_initialized and controller.is_pose_detector_ready
        
        # 收集调试信息，但不主动初始化任何组件
        debug_info = {}
        
        if is_initialized:
            if hasattr(controller, 'video_processor') and controller.video_processor:
                debug_info = controller.video_processor.get_debug_info()
        
        return jsonify({
            'success': True,
            'initialized': is_initialized,
            'running': is_running,
            'camera_ready': camera_ready,
            'pose_detector_ready': pose_detector_ready,
            'debug_info': debug_info,
            'status': 'ok' if (is_initialized and camera_ready) else 'error',
            'message': '系统正常运行' if (is_initialized and camera_ready) else '系统未就绪'
        })
    except Exception as e:
        import traceback
        current_app.logger.error(f"获取视频状态失败: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'status': 'error',
            'message': str(e),
            'debug_info': {'error': str(e), 'traceback': traceback.format_exc()}
        })

# 清理长期非活动客户端的定时任务
def cleanup_inactive_clients():
    """清理长期非活动的客户端"""
    inactive_threshold = 30.0  # 30秒
    current_time = time.time()
    clients_to_remove = []
    
    for client_id, client_info in _active_clients.items():
        if client_info['is_active'] and current_time - client_info['last_activity'] > inactive_threshold:
            print(f"客户端 {client_id} 超过 {inactive_threshold}秒没有活动，标记为非活跃")
            client_info['is_active'] = False
            clients_to_remove.append(client_id)
    
    # 移除非活跃客户端
    for client_id in clients_to_remove:
        cleanup_client(client_id)
    
    # 如果没有活跃客户端，可以考虑暂停视频处理
    active_count = sum(1 for client in _active_clients.values() if client['is_active'])
    if active_count == 0 and len(_active_clients) > 0:
        print("所有客户端已非活跃，清理历史连接记录")
        # 仅保留最近30个记录，避免字典过大
        if len(_active_clients) > 30:
            # 按最后活动时间排序
            sorted_clients = sorted(_active_clients.items(), key=lambda x: x[1]['last_activity'])
            # 仅保留最新的30个
            _active_clients.clear()
            for client_id, info in sorted_clients[-30:]:
                _active_clients[client_id] = info

# 注册蓝图
def register_video_routes(app):
    app.register_blueprint(video_bp)
    
    # 将控制器实例添加到应用扩展
    if 'rehab_controller' not in app.extensions:
        # 延迟导入，避免循环导入
        from app.modules.rehabilitation.controller import RehabilitationController
        controller = RehabilitationController()
        app.extensions['rehab_controller'] = controller
    
    # 启动定时清理任务
    def start_cleanup_task():
        while True:
            time.sleep(10)  # 每10秒运行一次
            try:
                cleanup_inactive_clients()
            except Exception as e:
                print(f"客户端清理任务错误: {str(e)}")
    
    # 在后台线程中启动清理任务
    cleanup_thread = threading.Thread(target=start_cleanup_task, daemon=True)
    cleanup_thread.start()
    
    print("视频流路由已注册") 