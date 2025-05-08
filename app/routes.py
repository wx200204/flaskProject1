# app/routes.py
import cv2
from flask import Blueprint, request, jsonify, current_app, send_file, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import uuid
import traceback
from .utils.processors import MedicalPreprocessor, PostureError
from .utils.analyzer import CobbAngleAnalyzer
from .utils.utils import save_report
from .utils.model_manager import ModelManager
from .utils.exporter import DataExporter
from .utils.monitor import SystemMonitor
import base64
import numpy as np
import time
import io

# 创建API蓝图
bp = Blueprint('api', __name__)  # 移除 url_prefix 以允许根路由处理

# 延迟初始化分析器 - 不在应用启动时加载模型
analyzer = None

# 创建系统监控实例
system_monitor = SystemMonitor()

@bp.route('/')
def index():
    # 重定向到main蓝图的index路由
    return redirect(url_for('main.index'))

@bp.route('/rehab')
def rehab_home():
    """康复主页"""
    return render_template('rehab/index.html')

@bp.route('/rehab/exercise/<exercise_id>')
def rehab_exercise(exercise_id):
    """康复训练页面"""
    # 根据训练ID获取训练详情
    exercise = current_app.rehab_manager.get_exercise_by_id(exercise_id)
    if not exercise:
        return render_template('rehab/error.html', message="找不到指定的训练项目")
    
    return render_template('rehab/exercise.html', exercise=exercise)

@bp.route('/rehab/session/<session_id>')
def rehab_session(session_id):
    """康复训练会话页面"""
    # 获取训练计划ID（在实际应用中应从数据库获取）
    plan_id = request.args.get('plan_id', 'default_plan')
    
    return render_template('rehab/session.html', 
                          session_id=session_id,
                          plan_id=plan_id)

@bp.route('/rehab/session/<session_id>/complete')
def rehab_session_complete(session_id):
    """康复训练会话完成页面"""
    return render_template('rehab/complete.html', session_id=session_id)

@bp.route('/rehab/error')
def rehab_error():
    """康复模块错误页面"""
    error_message = request.args.get('message', '发生未知错误')
    return render_template('rehab/error.html', error_message=error_message)

def get_model_manager():
    """获取模型管理器实例"""
    return current_app.model_manager

def get_analyzer():
    """获取分析器实例，如果不存在则创建一个"""
    global analyzer
    if analyzer is None:
        analyzer = CobbAngleAnalyzer()
    return analyzer

# API路由
@bp.route('/api/v1/status')
def status():
    return jsonify({
        'status': 'success',
        'message': 'API is running'
    })

@bp.route('/api/v1/analyze', methods=['POST'])
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
        
        # 尝试从请求中获取用户ID
        user_id = request.headers.get('X-User-ID') or request.args.get('user_id')
        
        # 从应用上下文获取模型管理器
        if hasattr(current_app, 'model_manager'):
            model_id = current_app.model_manager.get_active_model_id()
        else:
            model_id = 'default'
        
        # 导入数据库服务类
        from app.utils.database_service import DatabaseService
        
        # 记录分析结果到数据库
        if user_id:
            # 文件保存路径
            upload_dir = os.path.join(current_app.config['UPLOAD_DIR'], 'images')
            os.makedirs(upload_dir, exist_ok=True)
            file_id = str(uuid.uuid4())
            ext = '.jpg'
            filepath = os.path.join(upload_dir, f"{file_id}{ext}")
            
            # 保存图像
            with open(filepath, 'wb') as f:
                f.write(cv2.imencode('.jpg', image)[1])
                
            # 记录到数据库
            DatabaseService.record_analysis(
                user_id=user_id,
                image_path=filepath,
                cobb_angle=cobb_angle,
                results={
                    'cobb_angle': float(cobb_angle),
                    'confidence': float(confidence),
                    'severity': severity
                },
                model_id=model_id
            )
        
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

@bp.route('/api/v1/analyze/optimized', methods=['POST'])
def analyze_image_optimized():
    """优化版分析接口，使用增强算法"""
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
        start_time = time.time()
        current_app.logger.info(f"开始优化版图像分析: {file.filename}")
        
        # 读取上传的图像文件
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 图像基本检查
        h, w = image.shape[:2]
        if h < 300 or w < 300:
            return jsonify({
                'status': 'error',
                'message': '图像尺寸太小，无法进行可靠分析'
            }), 400
            
        # 检查图像方向
        if w > h:
            # 横向图像需要旋转为纵向
            current_app.logger.info("检测到横向图像，自动旋转")
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        # 获取原始图像的base64编码
        _, buffer = cv2.imencode('.jpg', image)
        original_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # 预处理图像 - 强化版处理
        preprocessor = MedicalPreprocessor(target_size=(768, 1024))  # 更高分辨率
        processed_image, preprocess_info = preprocessor.process(image)

        # 获取分析器并处理
        analyzer = get_analyzer()
        
        # 进行多种方法的分析比较
        methods_results = {}
        
        # 方法1: 标准分析
        try:
            standard_result = analyzer.analyze(processed_image)
            methods_results['standard'] = {
                'cobb_angle': standard_result.get('cobb_angle', 0.0),
                'confidence': standard_result.get('confidence', 0.0)
            }
        except Exception as e:
            current_app.logger.warning(f"标准分析失败: {str(e)}")
            methods_results['standard'] = {'error': str(e)}
        
        # 方法2: 使用改进版计算
        try:
            # 直接使用几何计算器的改进方法
            calculator = CobbAngleCalculator()
            
            if 'keypoints' in standard_result:
                keypoints = standard_result['keypoints']
                improved_img, improved_angle, improved_conf, improved_severity = calculator.calculate_improved(
                    processed_image, keypoints
                )
                
                methods_results['improved'] = {
                    'cobb_angle': improved_angle,
                    'confidence': improved_conf,
                    'severity': improved_severity
                }
                
                # 保存改进版结果图像
                result_image = improved_img
                cobb_angle = improved_angle
                confidence = improved_conf
                severity = improved_severity
            else:
                # 如果没有关键点，使用标准结果
                result_image = standard_result.get('result_image', processed_image.copy())
                cobb_angle = standard_result.get('cobb_angle', 0.0)
                confidence = standard_result.get('confidence', 0.0)
                severity = standard_result.get('severity', 'Unknown')
                
        except Exception as e:
            current_app.logger.error(f"改进版计算失败: {str(e)}")
            current_app.logger.error(traceback.format_exc())
            # 使用标准结果
            result_image = standard_result.get('result_image', processed_image.copy())
            cobb_angle = standard_result.get('cobb_angle', 0.0)
            confidence = standard_result.get('confidence', 0.0)
            severity = standard_result.get('severity', 'Unknown')
        
        # 编码结果图像
        _, buffer = cv2.imencode('.jpg', result_image)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 保存报告并添加更多信息
        report_path = save_report(image, result_image, cobb_angle, confidence, severity)
        
        # 构建结果，包含更丰富的信息
        result = {
            'status': 'success',
            'cobb_angle': float(cobb_angle),
            'confidence': float(confidence),
            'severity': severity,
            'original_image': original_image_base64,
            'result_image': result_image_base64,
            'report_url': f"/static/reports/{os.path.basename(report_path)}" if report_path else None,
            'analysis_time': time.time() - start_time,
            'preprocess_info': preprocess_info,
            'methods_comparison': methods_results,
            'timestamp': datetime.now().isoformat()
        }
        
        current_app.logger.info(f"图像分析完成, Cobb角: {cobb_angle:.2f}°, 置信度: {confidence:.2f}, "
                               f"分析耗时: {result['analysis_time']:.3f}秒")
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"优化版图像分析错误: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@bp.route('/api/v1/models', methods=['GET'])
def list_models():
    """获取模型列表"""
    model_manager = current_app.model_manager
    return jsonify(model_manager.list_models())

@bp.route('/api/v1/models/active', methods=['GET'])
def get_active_model():
    """获取当前激活的模型"""
    try:
        model_manager = get_model_manager()
        active_model = model_manager.get_active_model()
        if active_model:
            return jsonify({
                "code": 200,
                "data": active_model
            })
        else:
            return jsonify({"code": 4043, "msg": "No active model found"}), 404
    except Exception as e:
        current_app.logger.error(f"Get active model error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5007, "msg": "Failed to get active model"}), 500

@bp.route('/api/v1/models/<model_id>/activate', methods=['POST'])
def activate_model(model_id):
    """激活指定模型"""
    try:
        model_manager = get_model_manager()
        success = model_manager.activate_model(model_id)
        if success:
            # 重新初始化分析器以使用新模型
            global analyzer
            analyzer = CobbAngleAnalyzer(model_manager.get_model_path())
            
            return jsonify({
                "code": 200,
                "msg": "Model activated successfully"
            })
        else:
            return jsonify({"code": 4044, "msg": "Model not found"}), 404
    except Exception as e:
        current_app.logger.error(f"Activate model error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5008, "msg": f"Failed to activate model: {str(e)}"}), 500

@bp.route('/api/v1/models', methods=['POST'])
def upload_model():
    """上传新模型"""
    if 'file' not in request.files:
        return jsonify({"code": 4001, "msg": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"code": 4002, "msg": "Empty filename"}), 400
        
    # 获取模型信息
    name = request.form.get('name', 'Unnamed Model')
    description = request.form.get('description', '')
    version = request.form.get('version', '1.0')

    try:
        # 保存临时文件
        temp_path = os.path.join(current_app.config['UPLOAD_DIR'], f"temp_model_{uuid.uuid4()}.pth")
        file.save(temp_path)
        
        # 添加模型
        model_manager = get_model_manager()
        model_info = model_manager.add_model(temp_path, name, description, version)
        
        # 删除临时文件
        os.remove(temp_path)
        
        return jsonify({
            "code": 200,
            "data": model_info,
            "msg": "Model uploaded successfully"
        })
    except Exception as e:
        current_app.logger.error(f"Upload model error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5009, "msg": f"Failed to upload model: {str(e)}"}), 500

@bp.route('/api/v1/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """删除模型"""
    try:
        model_manager = get_model_manager()
        success = model_manager.delete_model(model_id)
        if success:
            return jsonify({
                "code": 200,
                "msg": "Model deleted successfully"
            })
        else:
            return jsonify({"code": 4044, "msg": "Model not found"}), 404
    except ValueError as ve:
        return jsonify({"code": 4003, "msg": str(ve)}), 400
    except Exception as e:
        current_app.logger.error(f"Delete model error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5010, "msg": f"Failed to delete model: {str(e)}"}), 500

@bp.route('/api/v1/models/<model_id>', methods=['GET'])
def get_model_info(model_id):
    """获取模型详细信息"""
    try:
        model_manager = get_model_manager()
        model_info = model_manager.get_model_info(model_id)
        if model_info:
            return jsonify({
                "code": 200,
                "data": model_info
            })
        else:
            return jsonify({"code": 4044, "msg": "Model not found"}), 404
    except Exception as e:
        current_app.logger.error(f"Get model info error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5011, "msg": "Failed to get model info"}), 500

@bp.route('/api/v1/export', methods=['GET'])
def export_data():
    """导出分析数据"""
    format_type = request.args.get('format', 'csv')
    
    try:
        exporter = DataExporter()
        
        if format_type == 'csv':
            output_path = exporter.export_to_csv()
            if output_path:
                return send_file(output_path, mimetype='text/csv', 
                                as_attachment=True, 
                                download_name=os.path.basename(output_path))
            else:
                return jsonify({"code": 4045, "msg": "No data to export"}), 404
                
        elif format_type == 'excel':
            # 由于pandas依赖问题，使用CSV替代Excel
            output_path = exporter.export_to_csv()
            if output_path:
                return send_file(output_path, mimetype='text/csv', 
                                as_attachment=True, 
                                download_name=os.path.basename(output_path).replace('.csv', '.csv'))
            else:
                return jsonify({"code": 4045, "msg": "No data to export"}), 404
                
        elif format_type == 'zip':
            output_path = exporter.export_all_data()
            if output_path:
                return send_file(output_path, mimetype='application/zip', 
                                as_attachment=True, 
                                download_name=os.path.basename(output_path))
            else:
                return jsonify({"code": 4045, "msg": "No data to export"}), 404
                
        else:
            return jsonify({"code": 4003, "msg": "Invalid export format"}), 400
            
    except Exception as e:
        current_app.logger.error(f"Export error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5012, "msg": f"Failed to export data: {str(e)}"}), 500

@bp.route('/api/v1/system/info', methods=['GET'])
def get_system_info():
    """获取系统信息"""
    try:
        info = system_monitor.get_system_info()
        return jsonify({
            "code": 200,
            "data": info
        })
    except Exception as e:
        current_app.logger.error(f"System info error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5013, "msg": "Failed to get system info"}), 500

@bp.route('/api/v1/system/usage', methods=['GET'])
def get_resource_usage():
    """获取资源使用情况"""
    try:
        usage = system_monitor.get_resource_usage()
        return jsonify({
            "code": 200,
            "data": usage
        })
    except Exception as e:
        current_app.logger.error(f"Resource usage error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5014, "msg": "Failed to get resource usage"}), 500

@bp.route('/api/v1/system/stats', methods=['GET'])
def get_app_stats():
    """获取应用统计信息"""
    try:
        stats = system_monitor.get_app_stats()
        return jsonify({
            "code": 200,
            "data": stats
        })
    except Exception as e:
        current_app.logger.error(f"App stats error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5015, "msg": "Failed to get app stats"}), 500

@bp.route('/api/v1/system/health', methods=['GET'])
def check_health():
    """系统健康检查"""
    try:
        health = system_monitor.check_health()
        status_code = 200
        if health["status"] == "critical":
            status_code = 500
        elif health["status"] == "warning":
            status_code = 429
            
        return jsonify({
            "code": status_code,
            "data": health
        }), status_code if status_code != 200 else 200
    except Exception as e:
        current_app.logger.error(f"Health check error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5016, "msg": "Failed to check health"}), 500

# 康复模块API
@bp.route('/api/v1/rehab/plan', methods=['POST'])
def create_rehab_plan():
    """创建康复方案API"""
    try:
        data = request.json
        
        # 验证必要参数
        if 'cobb_angle' not in data:
            return jsonify({"code": 4001, "msg": "缺少Cobb角度参数"}), 400
        
        # 获取参数
        cobb_angle = float(data['cobb_angle'])
        user_info = data.get('user_info', {})
        user_id = data.get('user_id', str(uuid.uuid4()))
        
        # 生成康复方案
        rehab_plan = current_app.rehab_manager.generate_rehab_plan(cobb_angle, user_info)
        
        return jsonify({
            "code": 200,
            "data": rehab_plan
        })
        
    except Exception as e:
        current_app.logger.error(f"创建康复方案失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5000, "msg": "创建康复方案失败"}), 500

@bp.route('/api/v1/rehab/session', methods=['POST'])
def create_rehab_session():
    """创建新的康复会话"""
    # 验证请求
    data = request.json
    if not data:
        return jsonify({
            'status': 'error',
            'message': '无效的请求数据'
        }), 400
    
    plan_id = data.get('plan_id', 'default_plan')
    user_id = data.get('user_id')
    
    try:
        # 获取康复管理器
        rehab_manager = current_app.rehab_manager if hasattr(current_app, 'rehab_manager') else None
        if not rehab_manager:
            return jsonify({
                'status': 'error',
                'message': '康复模块未初始化'
            }), 500
        
        # 创建会话
        session_id = rehab_manager.create_session(plan_id)
        
        # 导入数据库服务
        from app.utils.database_service import DatabaseService
        
        # 记录康复会话
        if user_id:
            session = DatabaseService.create_rehab_session(user_id, plan_id)
            session_id = session.session_id
        
        return jsonify({
            'status': 'success',
            'message': '成功创建康复会话',
            'data': {
                'session_id': session_id,
                'plan_id': plan_id
            }
        })
    except Exception as e:
        print(f"创建康复会话失败: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'创建康复会话失败: {str(e)}'
        }), 500

@bp.route('/api/v1/rehab/session/<session_id>', methods=['GET'])
def get_rehab_session(session_id):
    """获取康复会话信息API"""
    try:
        # 获取会话信息
        session = current_app.session_manager.get_session(session_id)
        
        if not session:
            return jsonify({"code": 4040, "msg": "找不到指定的康复会话"}), 404
        
        return jsonify({
            "code": 200,
            "data": session
        })
        
    except Exception as e:
        current_app.logger.error(f"获取康复会话失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5000, "msg": "获取康复会话失败"}), 500

@bp.route('/api/v1/rehab/session/<session_id>/exercise', methods=['GET'])
def get_current_exercise(session_id):
    """获取当前训练项目API"""
    try:
        # 获取当前训练项目
        exercise = current_app.session_manager.get_current_exercise(session_id)
        
        if not exercise:
            return jsonify({"code": 4040, "msg": "当前没有训练项目或会话不存在"}), 404
        
        return jsonify({
            "code": 200,
            "data": exercise
        })
        
    except Exception as e:
        current_app.logger.error(f"获取当前训练项目失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5000, "msg": "获取当前训练项目失败"}), 500

@bp.route('/api/v1/rehab/session/<session_id>/next', methods=['POST'])
def next_exercise(session_id):
    """移动到下一个训练项目API"""
    try:
        # 移动到下一个训练项目
        next_exercise = current_app.session_manager.move_to_next_exercise(session_id)
        
        if next_exercise:
            return jsonify({
                "code": 200,
                "data": {
                    "next_exercise": next_exercise,
                    "completed": False
                }
            })
        else:
            # 所有训练项目已完成
            return jsonify({
                "code": 200,
                "data": {
                    "next_exercise": None,
                    "completed": True,
                    "message": "所有训练项目已完成"
                }
            })
        
    except Exception as e:
        current_app.logger.error(f"移动到下一个训练项目失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5000, "msg": "移动到下一个训练项目失败"}), 500

@bp.route('/api/v1/rehab/session/<session_id>/evaluate', methods=['POST'])
def evaluate_pose(session_id):
    """评估姿势API"""
    try:
        if 'image' not in request.files:
            return jsonify({"code": 4001, "msg": "缺少图像文件"}), 400
        
        # 读取上传的图像
        file = request.files['image']
        if file.filename == '':
            return jsonify({"code": 4002, "msg": "未选择文件"}), 400
        
        # 读取图像数据
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 评估姿势
        result = current_app.session_manager.evaluate_pose(session_id, image)
        
        if "error" in result:
            return jsonify({"code": 5002, "msg": result["error"]}), 500
        
        # 转换图像为base64字符串，用于前端显示
        image_base64 = base64.b64encode(result["annotated_image"]).decode('utf-8')
        result["annotated_image"] = image_base64
        
        return jsonify({
            "code": 200,
            "data": result
        })
        
    except Exception as e:
        current_app.logger.error(f"评估姿势失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5000, "msg": "评估姿势失败"}), 500

@bp.route('/api/v1/rehab/session/<session_id>/end', methods=['POST'])
def end_session(session_id):
    """结束康复会话API"""
    try:
        # 结束会话
        success = current_app.session_manager.end_session(session_id)
        
        if not success:
            return jsonify({"code": 4040, "msg": "找不到指定的康复会话"}), 404
        
        return jsonify({
            "code": 200,
            "data": {
                "message": "会话已成功结束"
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"结束康复会话失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"code": 5000, "msg": "结束康复会话失败"}), 500

@bp.route('/api/rehab/reference_pose/<exercise_type>', methods=['GET'])
def get_reference_pose(exercise_type):
    """获取指定运动类型的参考姿势轮廓"""
    try:
        # 创建一个空白图像作为基础
        width, height = 640, 480
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 获取姿势检测器
        pose_detector = get_pose_detector()
        
        # 绘制参考姿势轮廓
        reference_image = pose_detector.draw_reference_pose(image, exercise_type)
        
        # 将图像编码为JPEG
        success, buffer = cv2.imencode('.jpg', reference_image)
        if not success:
            return jsonify({'error': '无法编码图像'}), 500
        
        # 保存到临时文件并返回URL
        temp_dir = os.path.join(current_app.static_folder, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        filename = f'reference_pose_{exercise_type}_{int(time.time())}.jpg'
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, 'wb') as f:
            f.write(buffer)
        
        # 返回图像URL
        return jsonify({
            'reference_image_url': url_for('static', filename=f'temp/{filename}')
        })
    except Exception as e:
        current_app.logger.error(f"生成参考姿势失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/rehab/detect_pose', methods=['POST'])
def detect_pose():
    """检测姿势并进行评估"""
    try:
        # 获取请求参数
        data = request.json
        exercise_type = data.get('exercise_type', 'spine_stretch')
        
        # 获取摄像头图像
        frame = get_camera_frame()
        if frame is None:
            return jsonify({'error': '无法获取摄像头图像'}), 400
        
        # 获取姿势检测器
        pose_detector = get_pose_detector()
        
        # 检测姿势
        landmarks, annotated_image = pose_detector.detect_pose(frame)
        
        # 评估姿势质量
        status, score, feedback = pose_detector.evaluate_pose_quality(landmarks, exercise_type)
        
        # 在图像上绘制参考姿势
        if landmarks is not None:
            annotated_image = pose_detector.draw_reference_pose(annotated_image, exercise_type, landmarks)
        
        # 保存标注后的图像
        success, buffer = cv2.imencode('.jpg', annotated_image)
        if not success:
            return jsonify({'error': '无法编码图像'}), 500
        
        # 保存到临时文件并返回URL
        temp_dir = os.path.join(current_app.static_folder, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        filename = f'pose_detection_{int(time.time())}.jpg'
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, 'wb') as f:
            f.write(buffer)
        
        # 返回评估结果和图像URL
        return jsonify({
            'status': 'success',
            'annotated_image_url': url_for('static', filename=f'temp/{filename}'),
            'evaluation': {
                'status': status.name,
                'score': float(score),
                'feedback': feedback
            }
        })
    except Exception as e:
        current_app.logger.error(f"姿势检测失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_pose_detector():
    """获取姿势检测器实例（单例模式）"""
    if not hasattr(current_app, 'pose_detector'):
        from app.models.rehab.pose_detector import PoseDetector
        current_app.pose_detector = PoseDetector()
    return current_app.pose_detector

def get_camera_frame():
    """获取摄像头帧（从Web摄像头流中）"""
    # 在实际应用中，这里应该从Web前端接收帧数据
    # 这里仅作为示例，使用临时图像代替
    
    # 尝试使用OpenCV从摄像头获取帧（仅用于测试）
    try:
        # 仅在测试模式下直接使用OpenCV
        if current_app.config.get('TESTING') or current_app.config.get('DEBUG'):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                current_app.logger.warning("无法打开摄像头")
                # 使用空白图像
                return np.ones((480, 640, 3), dtype=np.uint8) * 255
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                current_app.logger.warning("无法读取摄像头帧")
                return np.ones((480, 640, 3), dtype=np.uint8) * 255
            
            return frame
    except Exception as e:
        current_app.logger.error(f"获取摄像头帧失败: {str(e)}")
    
    # 默认情况下，使用空白图像
    return np.ones((480, 640, 3), dtype=np.uint8) * 255

# 康复训练会话API路由
@bp.route('/api/rehab/plan/<plan_id>', methods=['GET'])
def get_rehab_plan(plan_id):
    """获取康复训练计划详情"""
    try:
        # 获取康复管理器
        rehab_manager = get_rehab_manager()
        
        # 模拟根据ID获取计划（在实际应用中应从数据库获取）
        # 这里为了演示，直接创建一个新计划
        cobb_angle = 25  # 模拟从数据库获取的Cobb角度
        plan = rehab_manager.create_rehab_plan(cobb_angle)
        
        # 添加计划ID
        plan['plan_id'] = plan_id
        
        return jsonify(plan)
    except Exception as e:
        current_app.logger.error(f"获取康复计划失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/rehab/session/start', methods=['POST'])
def start_rehab_session():
    """开始康复训练会话"""
    try:
        data = request.json
        session_id = data.get('session_id')
        exercise_id = data.get('exercise_id')
        
        # 这里应该记录会话开始信息到数据库
        current_app.logger.info(f"开始康复训练会话: {session_id}, 训练项目: {exercise_id}")
        
        return jsonify({'status': 'success'})
    except Exception as e:
        current_app.logger.error(f"开始康复训练会话失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/rehab/session/complete', methods=['POST'])
def complete_rehab_exercise():
    """完成康复训练项目"""
    try:
        data = request.json
        session_id = data.get('session_id')
        exercise_id = data.get('exercise_id')
        duration = data.get('duration', 0)
        score = data.get('score', 0)
        
        # 导入数据库服务
        from app.utils.database_service import DatabaseService
        
        # 记录训练项目完成到数据库
        completion_rate = min(100, max(0, score))  # 确保分数在0-100之间
        exercise = DatabaseService.complete_rehab_exercise(session_id, exercise_id, completion_rate)
        
        current_app.logger.info(f"完成康复训练项目: {exercise_id}, 会话: {session_id}, 时长: {duration}秒, 得分: {score}")
        
        return jsonify({
            'status': 'success',
            'data': {
                'session_id': session_id,
                'exercise_id': exercise_id,
                'completion_rate': completion_rate
            }
        })
    except Exception as e:
        current_app.logger.error(f"完成康复训练项目失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/rehab/session/skip', methods=['POST'])
def skip_rehab_exercise():
    """跳过康复训练项目"""
    try:
        data = request.json
        session_id = data.get('session_id')
        exercise_id = data.get('exercise_id')
        
        # 这里应该记录跳过信息到数据库
        current_app.logger.info(f"跳过康复训练项目: {exercise_id}, 会话: {session_id}")
        
        return jsonify({'status': 'success'})
    except Exception as e:
        current_app.logger.error(f"跳过康复训练项目失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_rehab_manager():
    """获取康复管理器实例（单例模式）"""
    if not hasattr(current_app, 'rehab_manager'):
        from app.utils.rehab.rehab_manager import RehabManager
        current_app.rehab_manager = RehabManager()
    return current_app.rehab_manager

@bp.route('/api/rehab/start', methods=['POST'])
def start_rehab_session():
    """启动康复指导会话"""
    try:
        # 获取控制器实例
        from app.routes.video_stream import get_controller
        controller = get_controller()
        
        if not controller:
            return jsonify({
                'success': False,
                'message': '康复指导服务未初始化'
            })
        
        # 启动会话
        result = controller.start_session()
        
        # 统一返回格式
        if isinstance(result, dict) and 'status' in result:
            # 如果result已经是字典格式，转换status字段为success
            success = (result['status'] == 'success')
            response = {
                'success': success,
                'message': result.get('message', '操作完成'),
                'data': result
            }
        else:
            # 假设是布尔值或其他类型的结果
            success = bool(result)
            response = {
                'success': success,
                'message': '会话已启动' if success else '会话启动失败',
                'data': result
            }
            
        return jsonify(response)
    except Exception as e:
        import traceback
        current_app.logger.error(f"启动康复会话失败: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'message': f'服务器错误: {str(e)}'
        })

@bp.route('/api/rehab/stop', methods=['POST'])
def stop_rehab_session():
    """停止康复指导会话"""
    try:
        # 获取控制器实例
        from app.routes.video_stream import get_controller
        controller = get_controller()
        
        if not controller:
            return jsonify({
                'success': True,
                'message': '服务未运行'
            })
        
        # 停止会话
        result = controller.stop_session()
        
        # 统一返回格式
        if isinstance(result, dict) and 'status' in result:
            # 如果result已经是字典格式，转换status字段为success
            success = (result['status'] == 'success')
            response = {
                'success': success,
                'message': result.get('message', '操作完成'),
                'data': result
            }
        else:
            # 假设是布尔值或其他类型的结果
            success = bool(result)
            response = {
                'success': success,
                'message': '会话已停止' if success else '会话停止失败',
                'data': result
            }
            
        return jsonify(response)
    except Exception as e:
        import traceback
        current_app.logger.error(f"停止康复会话失败: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'message': f'服务器错误: {str(e)}'
        })

@bp.route('/video/keypoints', methods=['GET'])
def get_keypoints():
    """
    获取实时人体姿势关键点数据
    用于康复训练模块实时姿势检测和分析
    """
    try:
        # 获取姿势检测器
        pose_detector = get_pose_detector()
        
        # 获取摄像头图像
        frame = get_camera_frame()
        if frame is None or frame.size == 0:
            return jsonify({
                'detected': False,
                'message': '无法获取摄像头图像，请检查摄像头权限'
            })
        
        # 检测姿势
        landmarks, annotated_image = pose_detector.detect_pose(frame)
        
        if landmarks is None:
            return jsonify({
                'detected': False,
                'message': '未检测到人体，请确保您完全在画面中'
            })
        
        # 将关键点转换为前端可用的格式
        keypoints = pose_detector.get_keypoints_json()
        
        # 获取姿势评估结果 (假设当前进行的是脊柱伸展训练)
        status, score, feedback = pose_detector.evaluate_pose_quality(landmarks, "spine_stretch")
        
        # 返回关键点和评估数据
        return jsonify({
            'detected': True,
            'keypoints': keypoints,
            'score': float(score),
            'status': status.name,
            'feedback': feedback,
            'timestamp': time.time()
        })
        
    except Exception as e:
        current_app.logger.error(f"获取关键点数据失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'detected': False,
            'message': f'获取关键点数据失败: {str(e)}',
            'error': True
        })

@bp.route('/video/status', methods=['GET'])
def video_status():
    """检查视频处理服务状态"""
    try:
        # 检查姿势检测器是否已初始化
        detector_initialized = hasattr(current_app, 'pose_detector')
        
        # 尝试获取一帧来测试摄像头
        camera_ready = False
        try:
            frame = get_camera_frame()
            camera_ready = frame is not None and frame.size > 0
        except Exception as e:
            current_app.logger.error(f"摄像头测试失败: {str(e)}")
        
        status_info = {
            'status': 'ok' if detector_initialized and camera_ready else 'error',
            'initialized': detector_initialized,
            'camera_ready': camera_ready,
            'timestamp': time.time()
        }
        
        # 如果出错，添加更详细的错误信息
        if not detector_initialized:
            status_info['message'] = '姿势检测器未初始化'
        elif not camera_ready:
            status_info['message'] = '摄像头未准备就绪'
        
        return jsonify(status_info)
        
    except Exception as e:
        current_app.logger.error(f"检查视频状态失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': time.time()
        })