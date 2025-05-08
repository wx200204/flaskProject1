from flask import Blueprint, request, jsonify, current_app
from app.utils.decorators import api_error_handler, rate_limiter
import cv2
import numpy as np
import base64
import time
from datetime import datetime
import io
from PIL import Image
import os

# 创建蓝图，修改前缀为/api/rehab，与前端匹配
rehab_bp = Blueprint('rehabilitation', __name__, url_prefix='/api/rehab')

@rehab_bp.route('/start', methods=['POST'])
@api_error_handler
@rate_limiter(limit=5, period=60)  # 限制每分钟最多5次请求
def start_session():
    """启动康复指导会话"""
    controller = current_app.extensions.get('rehab_controller')
    if not controller:
        return jsonify({"status": "error", "message": "康复控制器未初始化"})
    return jsonify(controller.start_session())

@rehab_bp.route('/status', methods=['GET'])
@api_error_handler
def get_status():
    """获取当前康复会话状态和最新分析结果"""
    controller = current_app.extensions.get('rehab_controller')
    if not controller:
        return jsonify({"status": "error", "message": "康复控制器未初始化"})
    return jsonify(controller.get_latest_result())

@rehab_bp.route('/templates', methods=['GET'])
@api_error_handler
def get_templates():
    """获取可用的姿势模板"""
    controller = current_app.extensions.get('rehab_controller')
    if not controller:
        return jsonify({"status": "error", "message": "康复控制器未初始化"})
    return jsonify(controller.get_available_templates())

@rehab_bp.route('/template', methods=['POST'])
@api_error_handler
def change_template():
    """切换姿势模板"""
    controller = current_app.extensions.get('rehab_controller')
    if not controller:
        return jsonify({"status": "error", "message": "康复控制器未初始化"})
    
    data = request.get_json()
    if not data or 'template' not in data:
        return jsonify({"status": "error", "message": "缺少模板名称"})
    return jsonify(controller.change_template(data['template']))

@rehab_bp.route('/stop', methods=['POST'])
@api_error_handler
def stop_session():
    """停止康复指导会话"""
    controller = current_app.extensions.get('rehab_controller')
    if not controller:
        return jsonify({"status": "error", "message": "康复控制器未初始化"})
    return jsonify(controller.stop_session())

@rehab_bp.route('/detect_video_pose', methods=['POST'])
@api_error_handler
def detect_video_pose():
    """检测视频训练中的姿势并与模板进行比较"""
    try:
        # 获取控制器实例
        controller = current_app.extensions.get('rehab_controller')
        if not controller:
            return jsonify({
                "status": "ERROR",
                "message": "康复控制器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
            
        # 从请求中获取图像数据
        if 'image' not in request.files:
            return jsonify({
                "status": "ERROR",
                "message": "请求中未包含图像数据",
                "timestamp": datetime.now().timestamp()
            }), 400
            
        # 获取姿势类型
        pose_type = request.form.get('pose_type', 'standard')
        low_quality = request.form.get('low_quality', 'false').lower() == 'true'
        
        # 读取图像
        image_file = request.files['image']
        in_memory_file = io.BytesIO()
        image_file.save(in_memory_file)
        image_data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.COLOR_BGR2RGB)
        
        if image is None or image.size == 0:
            return jsonify({
                "status": "ERROR",
                "message": "无法解码图像数据",
                "timestamp": datetime.now().timestamp()
            }), 400
            
        # 使用视频处理器检测姿势
        pose_data = controller.video_processor.extract_pose_keypoints(image)
        
        # 修改检查逻辑以适应新的返回格式 (landmarks_array, annotated_image)
        landmarks, annotated_image = pose_data if pose_data else (None, None)
        
        if landmarks is None:
            return jsonify({
                "status": "NOT_DETECTED",
                "feedback": "未检测到人体姿势，请确保完整地出现在画面中",
                "timestamp": datetime.now().timestamp()
            })
        
        # 将新格式转换为旧格式，以便兼容后续代码
        pose_data = {
            'landmarks': landmarks,
            'annotated_frame': annotated_image
        }
        
        # 分析姿势角度
        angles = controller.posture_analyzer.calculate_spine_angles(pose_data['landmarks'])
        
        if not angles:
            return jsonify({
                "status": "NOT_DETECTED",
                "feedback": "无法计算脊柱角度，请确保您的躯干完全可见",
                "timestamp": datetime.now().timestamp()
            })
            
        # 与模板比较
        result = controller.posture_analyzer.compare_with_template(angles, pose_type)
        
        if not result:
            return jsonify({
                "status": "ERROR",
                "message": "姿势比对失败",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 扩展返回信息
        response_data = {
            "status": "SUCCESS",
            "poses": [{
                "keypoints": [
                    {
                        "id": i,
                        "x": landmark['x'],
                        "y": landmark['y'],
                        "z": landmark.get('z', 0),
                        "score": landmark.get('visibility', 1.0)
                    }
                    for i, landmark in enumerate(pose_data['landmarks'])
                ]
            }],
            "match_data": {
                "confidence": result["score"] / 100,  # 转换为0-1范围
                "feedback": result["feedback"],
                "duration": 0.0,  # 这个值会由前端计算
                "angles": angles
            },
            "timestamp": datetime.now().timestamp()
        }
        
        # 如果请求了低质量模式，简化返回的关键点数据
        if low_quality:
            # 只保留脊柱相关的关键点 (肩膀、髋部等)
            spine_indices = [11, 12, 23, 24]
            response_data["poses"][0]["keypoints"] = [
                point for point in response_data["poses"][0]["keypoints"]
                if point["id"] in spine_indices
            ]
        
        return jsonify(response_data)
        
    except Exception as e:
        current_app.logger.exception("姿势检测错误")
        return jsonify({
            "status": "ERROR",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_bp.route('/device/cameras', methods=['GET'])
@api_error_handler
def get_available_cameras():
    """获取可用摄像头列表"""
    try:
        # 获取控制器实例
        controller = current_app.extensions.get('rehab_controller')
        if not controller:
            return jsonify({
                "status": "ERROR",
                "message": "康复控制器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
            
        # 获取可用摄像头列表
        cameras = controller.video_processor.get_camera_options()
        
        return jsonify({
            "status": "SUCCESS",
            "cameras": cameras,
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception("获取摄像头列表错误")
        return jsonify({
            "status": "ERROR",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_bp.route('/device/switch-camera', methods=['POST'])
@api_error_handler
def switch_camera():
    """切换到指定摄像头"""
    try:
        # 获取控制器实例
        controller = current_app.extensions.get('rehab_controller')
        if not controller:
            return jsonify({
                "status": "ERROR",
                "message": "康复控制器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
            
        # 获取目标摄像头ID
        data = request.get_json()
        if not data or 'camera_id' not in data:
            return jsonify({
                "status": "ERROR",
                "message": "请求中未包含摄像头ID",
                "timestamp": datetime.now().timestamp()
            }), 400
            
        camera_id = data['camera_id']
        
        # 尝试切换摄像头
        success = controller.video_processor.switch_camera(camera_id)
        
        if not success:
            return jsonify({
                "status": "ERROR",
                "message": "切换摄像头失败",
                "timestamp": datetime.now().timestamp()
            }), 500
            
        return jsonify({
            "status": "SUCCESS",
            "message": f"已切换到摄像头 {camera_id}",
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception("切换摄像头错误")
        return jsonify({
            "status": "ERROR",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_bp.route('/reference_pose/<pose_type>', methods=['GET'])
@api_error_handler
def get_reference_pose(pose_type):
    """获取参考姿势图像"""
    try:
        # 首先尝试加载静态图片
        static_image_path = os.path.join(
            current_app.static_folder, 'img', 'rehab', f'{pose_type}_guide.svg'
        )
        
        if os.path.exists(static_image_path):
            # 读取SVG文件并返回
            with open(static_image_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
                
            return jsonify({
                "status": "SUCCESS",
                "image_format": "svg+xml",
                "image_base64": f"data:image/svg+xml;base64,{base64.b64encode(svg_content.encode()).decode()}",
                "timestamp": datetime.now().timestamp()
            })
        
        # 如果没有静态图片，尝试动态生成
        # 获取控制器实例
        controller = current_app.extensions.get('rehab_controller')
        if not controller:
            return jsonify({
                "status": "ERROR",
                "message": "康复控制器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
            
        # 尝试获取参考姿势数据
        template = controller.posture_analyzer.posture_templates.get(pose_type)
        if not template:
            # 使用标准直立姿势作为后备
            template = controller.posture_analyzer.posture_templates.get("标准直立姿势")
            
        if not template:
            return jsonify({
                "status": "ERROR",
                "message": "未找到参考姿势模板",
                "timestamp": datetime.now().timestamp()
            }), 404
            
        # 创建空白图像
        blank_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 在图像上绘制简单的指导文本
        cv2.putText(blank_image, 
                   f"姿势指南: {template.get('name', pose_type)}", 
                   (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                   
        cv2.putText(blank_image, 
                   f"{template.get('description', '请参照视频保持正确姿势')}", 
                   (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                   
        # 将图像编码为JPEG
        _, buffer = cv2.imencode('.jpg', blank_image)
        jpg_base64 = base64.b64encode(buffer).decode()
        
        return jsonify({
            "status": "SUCCESS",
            "image_format": "jpeg",
            "image_base64": f"data:image/jpeg;base64,{jpg_base64}",
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception("获取参考姿势图像错误")
        return jsonify({
            "status": "ERROR",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_bp.route('/mp/connect', methods=['POST'])
@api_error_handler
def mp_connect():
    """微信小程序连接初始化接口"""
    controller = current_app.extensions.get('rehab_controller')
    if not controller:
        return jsonify({"status": "error", "message": "康复控制器未初始化"})
    
    data = request.get_json() or {}
    client_id = data.get('client_id', f"mp_client_{int(time.time())}")
    device_info = data.get('device_info', {})
    
    # 记录小程序客户端信息
    client_info = {
        'client_id': client_id,
        'device_info': device_info,
        'connect_time': datetime.now().isoformat(),
        'last_active': datetime.now().isoformat()
    }
    
    # 存储客户端信息（可以保存到应用上下文或数据库）
    if not hasattr(current_app, 'mp_clients'):
        current_app.mp_clients = {}
    current_app.mp_clients[client_id] = client_info
    
    return jsonify({
        "status": "success",
        "client_id": client_id,
        "server_time": datetime.now().isoformat(),
        "templates": controller.get_available_templates().get('templates', [])
    })

@rehab_bp.route('/mp/analyze', methods=['POST'])
@api_error_handler
def mp_analyze_pose():
    """小程序康复姿势分析API"""
    try:
        # 获取控制器实例
        controller = current_app.extensions.get('rehab_controller')
        if not controller:
            return jsonify({
                "status": "ERROR",
                "message": "康复控制器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 获取请求数据
        data = request.get_json()
        if not data or 'keypoints' not in data or 'template_id' not in data:
            return jsonify({
                "status": "ERROR",
                "message": "请求缺少必要数据",
                "timestamp": datetime.now().timestamp()
            }), 400
        
        # 获取模板管理器
        template_manager = current_app.extensions.get('rehab_template_manager')
        if not template_manager:
            return jsonify({
                "status": "ERROR", 
                "message": "模板管理器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 获取关键点和模板ID
        keypoints = data['keypoints']
        template_id = data['template_id']
        frame_size = data.get('frame_size')  # 可选参数
        
        # 获取模板
        template = template_manager.get_template(template_id)
        if not template:
            return jsonify({
                "status": "ERROR",
                "message": f"模板 {template_id} 不存在",
                "timestamp": datetime.now().timestamp()
            }), 404
        
        # 匹配姿势
        score, feedback = template.match_pose(keypoints, frame_size)
        
        # 返回匹配结果
        return jsonify({
            "status": "SUCCESS",
            "result": {
                "score": score,
                "feedback": feedback,
                "template_id": template_id,
                "template_name": template.name,
                "timestamp": datetime.now().timestamp()
            }
        })
        
    except Exception as e:
        current_app.logger.exception("小程序康复姿势分析失败")
        return jsonify({
            "status": "ERROR",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_bp.route('/mp/session', methods=['POST'])
@api_error_handler
def mp_session_control():
    """微信小程序康复会话控制接口"""
    controller = current_app.extensions.get('rehab_controller')
    if not controller:
        return jsonify({"status": "error", "message": "康复控制器未初始化"})
    
    data = request.get_json() or {}
    action = data.get('action')
    client_id = data.get('client_id')
    
    if not action:
        return jsonify({"status": "error", "message": "未指定操作类型"})
    
    # 验证客户端
    if hasattr(current_app, 'mp_clients') and client_id in current_app.mp_clients:
        # 更新最后活动时间
        current_app.mp_clients[client_id]['last_active'] = datetime.now().isoformat()
    
    # 处理不同的会话操作
    if action == 'start':
        template_name = data.get('template', '标准直立姿势')
        controller.template = template_name
        result = controller.start_session()
        return jsonify(result)
    
    elif action == 'stop':
        result = controller.stop_session()
        return jsonify(result)
    
    elif action == 'change_template':
        template_name = data.get('template')
        if not template_name:
            return jsonify({"status": "error", "message": "未指定模板名称"})
        result = controller.change_template(template_name)
        return jsonify(result)
    
    else:
        return jsonify({"status": "error", "message": f"不支持的操作: {action}"})

@rehab_bp.route('/mp/progress', methods=['GET', 'POST'])
@api_error_handler
def mp_get_progress():
    """获取微信小程序康复训练进度"""
    controller = current_app.extensions.get('rehab_controller')
    if not controller:
        return jsonify({"status": "error", "message": "康复控制器未初始化"})
    
    # 获取客户端ID和会话ID
    client_id = request.args.get('client_id') or (request.get_json() or {}).get('client_id')
    session_id = request.args.get('session_id') or (request.get_json() or {}).get('session_id')
    
    # 验证客户端
    if hasattr(current_app, 'mp_clients') and client_id in current_app.mp_clients:
        # 更新最后活动时间
        current_app.mp_clients[client_id]['last_active'] = datetime.now().isoformat()
    
    # 获取训练进度数据
    progress_data = controller.get_progress_metrics(session_id)
    
    # 添加训练历史图表数据
    if progress_data["status"] == "success" and controller.analysis_results:
        # 抽样最多20个数据点，避免数据过多
        sample_step = max(1, len(controller.analysis_results) // 20)
        sampled_results = controller.analysis_results[::sample_step]
        
        # 构建图表数据
        chart_data = {
            "timestamps": [],
            "scores": [],
            "labels": []
        }
        
        for i, result in enumerate(sampled_results):
            if isinstance(result, dict) and "score" in result:
                # 添加时间戳（相对于开始时间）
                relative_time = i * sample_step  # 简化的时间表示
                chart_data["timestamps"].append(relative_time)
                
                # 添加分数
                chart_data["scores"].append(result["score"])
                
                # 添加标签
                chart_data["labels"].append(result.get("status", ""))
        
        progress_data["chart_data"] = chart_data
    
    # 添加对比数据（如果有历史会话）
    if hasattr(current_app, 'rehab_history') and current_app.rehab_history:
        recent_sessions = list(current_app.rehab_history.values())[-5:]  # 最近5个会话
        
        comparison_data = {
            "sessions": [],
            "avg_scores": []
        }
        
        for session in recent_sessions:
            comparison_data["sessions"].append(session.get("session_id", "未知会话"))
            comparison_data["avg_scores"].append(session.get("avg_score", 0))
        
        progress_data["comparison_data"] = comparison_data
    
    return jsonify(progress_data)
    
@rehab_bp.route('/mp/templates', methods=['GET'])
@api_error_handler
def mp_get_templates():
    """小程序获取康复模板列表"""
    try:
        # 获取模板管理器
        template_manager = current_app.extensions.get('rehab_template_manager')
        if not template_manager:
            return jsonify({
                "status": "ERROR", 
                "message": "模板管理器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 获取查询参数
        cobb_angle = request.args.get('cobb_angle', type=float)
        template_type = request.args.get('type')
        
        # 根据Cobb角度确定严重程度
        severity_level = None
        if cobb_angle:
            if 10 <= cobb_angle < 25:
                severity_level = 0  # 轻度
            elif 25 <= cobb_angle < 40:
                severity_level = 1  # 中度
            elif cobb_angle >= 40:
                severity_level = 2  # 重度
        
        # 根据查询参数获取模板
        if template_type and severity_level is not None:
            templates = template_manager.get_templates_by_type(template_type, severity_level)
        elif template_type:
            templates = template_manager.get_templates_by_type(template_type)
        elif severity_level is not None:
            templates = template_manager.get_templates_by_severity(severity_level)
        else:
            templates = template_manager.get_all_templates()
        
        # 为小程序格式化模板数据
        template_list = []
        for template in templates:
            template_data = template.save_to_json()
            # 添加模板描述
            template_data["description"] = self._get_template_description(template)
            template_list.append(template_data)
        
        return jsonify({
            "status": "SUCCESS",
            "templates": template_list,
            "count": len(template_list),
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception("获取康复模板列表失败")
        return jsonify({
            "status": "ERROR",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_bp.route('/mp/recommended-templates', methods=['GET'])
@api_error_handler
def mp_get_recommended_templates():
    """小程序获取推荐康复模板"""
    try:
        # 获取模板管理器
        template_manager = current_app.extensions.get('rehab_template_manager')
        if not template_manager:
            return jsonify({
                "status": "ERROR", 
                "message": "模板管理器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 获取Cobb角度参数
        cobb_angle = request.args.get('cobb_angle', type=float)
        if not cobb_angle:
            return jsonify({
                "status": "ERROR",
                "message": "缺少必要参数: cobb_angle",
                "timestamp": datetime.now().timestamp()
            }), 400
        
        # 根据Cobb角度确定严重程度
        severity_level = None
        if 10 <= cobb_angle < 25:
            severity_level = 0  # 轻度
        elif 25 <= cobb_angle < 40:
            severity_level = 1  # 中度
        elif cobb_angle >= 40:
            severity_level = 2  # 重度
        else:
            return jsonify({
                "status": "ERROR",
                "message": f"Cobb角度 {cobb_angle} 超出有效范围",
                "timestamp": datetime.now().timestamp()
            }), 400
        
        # 获取该严重程度的所有模板
        templates = template_manager.get_templates_by_severity(severity_level)
        
        # 为小程序格式化模板数据并按类型分组
        template_groups = {}
        for template in templates:
            template_type = template.template_type.value if hasattr(template.template_type, 'value') else template.template_type
            
            if template_type not in template_groups:
                template_groups[template_type] = []
            
            template_data = template.save_to_json()
            # 添加模板描述
            template_data["description"] = self._get_template_description(template)
            template_groups[template_type].append(template_data)
        
        # 构建推荐方案
        recommended_plan = {
            "cobb_angle": cobb_angle,
            "severity_level": severity_level,
            "severity_name": ["轻度", "中度", "重度"][severity_level],
            "template_groups": [
                {
                    "type": type_id,
                    "name": self._get_template_type_name(type_id),
                    "templates": templates
                }
                for type_id, templates in template_groups.items()
            ],
            "daily_recommendation": {
                "minutes": 30,
                "sessions": 2,
                "notes": "建议每日进行2次康复训练，每次约15分钟，请在训练前热身，训练后放松。"
            }
        }
        
        return jsonify({
            "status": "SUCCESS",
            "recommended_plan": recommended_plan,
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception("获取推荐康复模板失败")
        return jsonify({
            "status": "ERROR",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

def _get_template_description(self, template):
    """生成模板描述"""
    template_type = template.template_type.value if hasattr(template.template_type, 'value') else template.template_type
    severity = template.severity_level.value if hasattr(template.severity_level, 'value') else template.severity_level
    
    severity_names = ["轻度", "中度", "重度"]
    severity_name = severity_names[severity] if 0 <= severity < len(severity_names) else "未知"
    
    type_descriptions = {
        "spine_stretch": "脊柱伸展",
        "side_bend": "侧弯拉伸",
        "rotation": "躯干旋转",
        "lying": "卧姿康复",
        "custom": "自定义康复"
    }
    
    type_name = type_descriptions.get(template_type, "康复动作")
    
    return f"{severity_name}脊柱侧弯{type_name}训练，帮助改善脊柱姿态，增强肌肉力量。"

def _get_template_type_name(self, type_id):
    """获取模板类型名称"""
    type_names = {
        "spine_stretch": "脊柱伸展训练",
        "side_bend": "侧弯拉伸训练",
        "rotation": "躯干旋转训练",
        "lying": "卧姿康复训练",
        "custom": "自定义康复训练"
    }
    
    return type_names.get(type_id, "康复训练")

@rehab_bp.route('/mp/save-session', methods=['POST'])
@api_error_handler
def mp_save_session():
    """保存康复训练会话数据"""
    controller = current_app.extensions.get('rehab_controller')
    if not controller:
        return jsonify({"status": "error", "message": "康复控制器未初始化"})
    
    # 获取请求数据
    data = request.get_json() or {}
    client_id = data.get('client_id')
    
    # 验证会话是否活跃
    if not controller.is_running:
        return jsonify({"status": "error", "message": "当前没有活跃的康复会话"})
    
    # 获取会话数据
    progress_data = controller.get_progress_metrics()
    if progress_data["status"] != "success":
        return jsonify({"status": "error", "message": "无法获取会话数据"})
    
    # 构建会话摘要
    session_summary = {
        "session_id": controller.session_id,
        "client_id": client_id,
        "start_time": datetime.fromtimestamp(controller.start_time).isoformat() if controller.start_time else None,
        "end_time": datetime.now().isoformat(),
        "duration": progress_data["session_info"]["duration"],
        "template": controller.template,
        "avg_score": progress_data["metrics"]["avg_score"],
        "min_score": progress_data["metrics"]["min_score"],
        "max_score": progress_data["metrics"]["max_score"],
        "stability": progress_data["metrics"]["stability"],
        "samples": progress_data["metrics"]["samples"],
        "status_distribution": progress_data["metrics"]["status_distribution"]
    }
    
    # 保存会话摘要
    if not hasattr(current_app, 'rehab_history'):
        current_app.rehab_history = {}
    
    current_app.rehab_history[controller.session_id] = session_summary
    
    # 返回摘要
    return jsonify({
        "status": "success", 
        "message": "会话数据已保存",
        "summary": session_summary
    })

# 注册蓝图的函数
def register_rehabilitation_routes(app):
    # 确保控制器实例已存在
    if 'rehab_controller' not in app.extensions:
        from app.modules.rehabilitation.controller import RehabilitationController
        controller = RehabilitationController()
        app.extensions['rehab_controller'] = controller
        print("创建了新的康复指导控制器实例")
    
    # 注册蓝图
    app.register_blueprint(rehab_bp)
    
    # 打印所有注册的路由
    print("\n=== 康复指导API路由 ===")
    for rule in app.url_map.iter_rules():
        if rule.endpoint.startswith('rehabilitation.'):
            print(f"路由: {rule}, 端点: {rule.endpoint}, 方法: {rule.methods}")
    print("=== 康复指导路由已注册 ===\n") 