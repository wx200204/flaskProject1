from flask import Blueprint, request, jsonify, current_app, g
import cv2
import numpy as np
import base64
import os
from datetime import datetime
import traceback
from ..utils.processors import MedicalPreprocessor
from ..utils.analyzer import CobbAngleAnalyzer
from ..utils.utils import save_report
from app.utils.database_service import DatabaseService
import random

# 创建API蓝图
api_bp = Blueprint('api', __name__)

# 获取分析器
def get_analyzer():
    """获取分析器实例，如果不存在则创建一个"""
    if not hasattr(current_app, '_analyzer'):
        current_app._analyzer = CobbAngleAnalyzer()
    return current_app._analyzer

@api_bp.route('/api/v1/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file uploaded'
        }), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        }), 400

    try:
        # 读取上传的图像文件
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 记录原始图像信息
        current_app.logger.info(f"接收到图像: {image.shape}, 类型: {image.dtype}")
        
        # 获取原始图像的base64编码
        _, buffer = cv2.imencode('.jpg', image)
        original_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # 预处理图像
        preprocessor = MedicalPreprocessor()
        processed_image, preprocess_info = preprocessor.process(image)

        # 获取分析器
        analyzer = get_analyzer()
        
        # 分析图像 - 使用改进版算法
        current_app.logger.info(f"分析处理后的图像: {processed_image.shape}")
        try:
            # 尝试使用改进版算法
            analysis_result = analyzer.analyze_optimized(processed_image)
            current_app.logger.info("使用改进版分析算法处理图像")
        except Exception as e:
            # 如果改进版失败，回退到原始算法
            current_app.logger.warning(f"改进版分析失败，回退到原始算法: {str(e)}")
            analysis_result = analyzer.analyze(processed_image)
        
        # 确保结果图像存在并且有效
        if 'result_image' not in analysis_result or analysis_result['result_image'] is None:
            current_app.logger.warning("结果中没有有效的result_image")
            # 创建一个基本的结果图像
            result_image = processed_image.copy()
            cv2.putText(result_image, "未生成结果图像", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # 添加Cobb角度信息
            cobb_angle = analysis_result.get('cobb_angle', 0.0)
            cv2.putText(result_image, f"Cobb角: {cobb_angle:.1f}°", 
                       (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            analysis_result['result_image'] = result_image
                
        # 编码结果图像
        result_image = analysis_result['result_image']
        _, buffer = cv2.imencode('.jpg', result_image)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 提取分析结果
        cobb_angle = analysis_result.get('cobb_angle', 0.0)
        confidence = analysis_result.get('confidence', 0.0)
        severity = analysis_result.get('severity', "Unknown")
        
        # 保存报告 (可选)
        report_path = save_report(image, result_image, cobb_angle, confidence, severity)
        
        # 返回结果
        result = {
            'status': 'success',
            'cobb_angle': float(cobb_angle),
            'confidence': float(confidence),
            'severity': severity,
            'original_image': original_image_base64,
            'result_image': result_image_base64,
            'report_url': f"/static/reports/{os.path.basename(report_path)}" if report_path else None,
            'preprocess_info': preprocess_info,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Image analysis error: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 

@api_bp.route('/api/v1/user/register', methods=['POST'])
def register_user():
    """注册用户"""
    data = request.json
    username = data.get('username')
    email = data.get('email')
    
    if not username:
        return jsonify({
            'status': 'error',
            'message': '用户名不能为空'
        }), 400
    
    # 创建用户
    user = DatabaseService.get_or_create_user(username, email)
    
    return jsonify({
        'status': 'success',
        'message': '用户创建成功',
        'data': {
            'user_id': user.id,
            'username': user.username
        }
    })

@api_bp.route('/api/v1/user/records', methods=['GET'])
def get_user_records():
    """获取用户记录"""
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({
            'status': 'error',
            'message': '用户ID不能为空'
        }), 400
    
    # 获取用户分析记录
    analysis_records = DatabaseService.get_user_analysis_records(user_id)
    
    # 转换为JSON格式
    records = []
    for record in analysis_records:
        records.append({
            'id': record.id,
            'image_path': record.image_path,
            'cobb_angle': record.cobb_angle,
            'model_id': record.model_id,
            'created_at': record.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return jsonify({
        'status': 'success',
        'data': {
            'records': records
        }
    })

@api_bp.route('/api/v1/user/rehab_sessions', methods=['GET'])
def get_user_rehab_sessions():
    """获取用户康复训练会话"""
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({
            'status': 'error',
            'message': '用户ID不能为空'
        }), 400
    
    # 获取用户康复训练会话
    rehab_sessions = DatabaseService.get_user_rehab_sessions(user_id)
    
    # 转换为JSON格式
    sessions = []
    for session in rehab_sessions:
        # 获取会话中的训练项目
        exercises = DatabaseService.get_session_exercises(session.session_id)
        
        # 转换训练项目
        exercise_list = []
        for exercise in exercises:
            exercise_list.append({
                'id': exercise.id,
                'exercise_id': exercise.exercise_id,
                'exercise_name': exercise.exercise_name,
                'start_time': exercise.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': exercise.end_time.strftime('%Y-%m-%d %H:%M:%S') if exercise.end_time else None,
                'duration': exercise.duration,
                'completion_rate': exercise.completion_rate,
                'status': exercise.status
            })
        
        sessions.append({
            'id': session.id,
            'session_id': session.session_id,
            'plan_id': session.plan_id,
            'start_time': session.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': session.end_time.strftime('%Y-%m-%d %H:%M:%S') if session.end_time else None,
            'status': session.status,
            'exercises': exercise_list
        })
    
    return jsonify({
        'status': 'success',
        'data': {
            'sessions': sessions
        }
    })

@api_bp.route('/api/v1/rehab/session/<session_id>/result', methods=['GET'])
def get_rehab_session_result(session_id):
    """获取康复训练会话结果"""
    try:
        # 在实际应用中应该从数据库获取会话结果
        # 这里模拟返回结果数据
        
        # 生成随机结果数据
        total_score = random.randint(70, 95)
        total_time = random.randint(300, 900)  # 总秒数
        total_exercises = 3
        completed_exercises = random.randint(2, 3)
        
        # 根据得分生成反馈
        feedback = ""
        if total_score >= 90:
            feedback = "您的训练表现非常出色！继续保持这种高质量的训练将显著改善您的脊柱健康。"
        elif total_score >= 80:
            feedback = "您的训练表现良好，动作准确度高。坚持训练将帮助您改善脊柱状况。"
        elif total_score >= 70:
            feedback = "您的训练表现不错，但还有提升空间。注意动作的准确性和完整性。"
        else:
            feedback = "您完成了训练，但动作准确度有待提高。建议关注训练指导，提升动作质量。"
            
        result_data = {
            "session_id": session_id,
            "total_score": total_score,
            "total_time": total_time,
            "total_exercises": total_exercises,
            "completed_exercises": completed_exercises,
            "feedback": feedback,
            "completed_at": datetime.now().isoformat()
        }
        
        return jsonify({
            "code": 200,
            "data": result_data,
            "msg": "获取会话结果成功"
        })
        
    except Exception as e:
        current_app.logger.error(f"获取康复会话结果错误: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({"code": 500, "msg": f"服务器错误: {str(e)}"}), 500

# 在请求前执行，用于设置用户ID
@api_bp.before_request
def before_request():
    # 从请求头、会话或查询参数中获取用户ID
    user_id = request.headers.get('X-User-ID') or request.args.get('user_id')
    if user_id:
        g.user_id = user_id 