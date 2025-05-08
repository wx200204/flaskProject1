from flask import Blueprint, render_template, request, jsonify, current_app
import cv2
import numpy as np
import base64
import os
from datetime import datetime
import traceback
from ..models.posture_analyzer import PostureAnalyzer
from werkzeug.utils import secure_filename
import time
import json

def allowed_file(filename):
    """检查文件是否具有允许的扩展名"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config.get('ALLOWED_EXTENSIONS', ['jpg', 'jpeg', 'png'])

# 用于将NumPy类型转换为Python原生类型，以便于JSON序列化
def convert_numpy_to_python(obj):
    """递归地将NumPy类型转换为Python原生类型
    
    Args:
        obj: 需要转换的对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

# 创建体态评估蓝图
posture_bp = Blueprint('posture_eval', __name__, url_prefix='/posture')

# 获取体态分析器
def get_posture_analyzer():
    """获取体态分析器实例，如果不存在则创建一个"""
    if not hasattr(current_app, '_posture_analyzer'):
        current_app._posture_analyzer = PostureAnalyzer()
    return current_app._posture_analyzer

@posture_bp.route('/')
def posture_index():
    """体态评估首页"""
    return render_template('posture_evaluation.html')

@posture_bp.route('/evaluate', methods=['POST'])
def evaluate_posture():
    """评估用户体态，支持上传部分或全部视图图片
    
    支持上传的视图:
        - 前视图(front)
        - 后视图(back)
        - 左侧视图(left)
        - 右侧视图(right)
    
    如果用户上传了完整的四视图，将进行更高精度的综合分析
    
    返回:
        JSON格式的体态分析结果
    """
    try:
        # 检查是否有上传的文件
        if 'front' not in request.files and 'back' not in request.files and 'left' not in request.files and 'right' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '请至少上传一个视图的照片'
            })
        
        # 处理上传的文件
        images = {}
        uploaded_views = []
        missing_views = []
        
        for view in ['front', 'back', 'left', 'right']:
            if view in request.files and request.files[view].filename != '':
                file = request.files[view]
                
                # 文件格式验证
                if not allowed_file(file.filename):
                    return jsonify({
                        'status': 'error',
                        'message': f'不支持的文件格式。请上传 {", ".join(current_app.config["ALLOWED_EXTENSIONS"])} 格式的图片'
                    })
                
                # 保存文件
                filename = secure_filename(file.filename)
                timestamp = int(time.time())
                new_filename = f"{view}_{timestamp}_{filename}"
                filepath = os.path.join(current_app.config['UPLOAD_DIR'], new_filename)
                file.save(filepath)
                
                # 读取图像
                image = cv2.imread(filepath)
                if image is None:
                    return jsonify({
                        'status': 'error',
                        'message': f'无法读取{view}视图图片'
                    })
                
                images[view] = image
                uploaded_views.append(view)
            else:
                missing_views.append(view)
        
        # 确保至少有一个视图
        if not images:
            return jsonify({
                'status': 'error',
                'message': '请至少上传一个视图的照片'
            })
        
        # 初始化体态分析器
        analyzer = PostureAnalyzer()
        
        # 分析体态
        result = analyzer.analyze_posture(images)
        
        # 构建视图消息
        view_labels = {
            'front': '正面视图',
            'back': '背面视图',
            'left': '左侧视图',
            'right': '右侧视图'
        }
        
        missing_view_labels = [view_labels[view] for view in missing_views]
        
        # 编码可视化图像为base64
        visualizations = {}
        for view, img in result.get('visualization', {}).items():
            if img is not None:
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                visualizations[view] = f"data:image/jpeg;base64,{img_base64}"
        
        # 添加综合分析标志和说明
        is_comprehensive = result.get('is_comprehensive', False)
        analysis_quality_message = ""
        
        if is_comprehensive:
            analysis_quality_message = "完整四视图分析提供了高可靠性的综合评估结果。"
        elif len(uploaded_views) > 1:
            analysis_quality_message = f"已分析{len(uploaded_views)}个视图，提供了部分综合评估结果。"
        else:
            analysis_quality_message = "仅分析了单一视角，评估结果可能不够全面。"
        
        # 构建响应
        response = {
            'status': 'success',
            'analyzed_views': uploaded_views,
            'missing_views': missing_views,
            'posture_issues': result.get('posture_issues', []),
            'severity': result.get('severity', '正常'),
            'recommendations': result.get('recommendations', ''),
            'angles': result.get('angles', {}),
            'measurements': result.get('measurements', {}),
            'assessment': result.get('assessment', {}),
            'visualizations': visualizations,
            'is_comprehensive': is_comprehensive,
            'analysis_quality_message': analysis_quality_message
        }
        
        # 如果有缺失的视图，添加提示消息
        if missing_views:
            if len(missing_views) < 4:
                # 部分视图缺失
                response['missing_views_message'] = f"未提供{', '.join(missing_view_labels)}照片，相关体态问题可能无法完全分析。"
            else:
                # 全部视图缺失（理论上不会发生，因为之前已检查至少有一个视图）
                response['missing_views_message'] = "未提供任何视图照片，无法进行体态分析。"
        
        # 转换NumPy类型为Python原生类型，以便JSON序列化
        response = convert_numpy_to_python(response)
                
        return jsonify(response)
        
    except Exception as e:
        current_app.logger.error(f"体态评估失败: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'体态评估失败: {str(e)}'
        })

def register_posture_routes(app):
    """注册体态评估相关路由"""
    # 检查是否已经注册，避免重复注册
    if 'posture_eval' not in app.blueprints:
        app.register_blueprint(posture_bp) 