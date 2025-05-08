from flask import Blueprint, request, jsonify, current_app
from app.utils.decorators import api_error_handler, rate_limiter
from app.models.rehab.rehab_template import RehabTemplateManager, RehabTemplateType, SeverityLevel, RehabTemplate
import json
import os
from datetime import datetime

# 创建蓝图，前缀为/api/rehab-template
rehab_template_bp = Blueprint('rehab_template', __name__, url_prefix='/api/rehab-template')

@rehab_template_bp.route('/list', methods=['GET'])
@api_error_handler
def get_templates():
    """获取所有康复模板"""
    try:
        # 获取模板管理器
        template_manager = current_app.extensions.get('rehab_template_manager')
        if not template_manager:
            return jsonify({
                "status": "error", 
                "message": "模板管理器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 获取查询参数
        template_type = request.args.get('type')
        severity = request.args.get('severity')
        
        # 根据查询参数获取模板
        if template_type and severity:
            # 转换severity为枚举值
            severity_level = None
            if severity.isdigit():
                severity_level = int(severity)
            elif severity.lower() == "mild":
                severity_level = SeverityLevel.MILD.value
            elif severity.lower() == "moderate":
                severity_level = SeverityLevel.MODERATE.value
            elif severity.lower() == "severe":
                severity_level = SeverityLevel.SEVERE.value
                
            templates = template_manager.get_templates_by_type(template_type, severity_level)
        elif template_type:
            templates = template_manager.get_templates_by_type(template_type)
        elif severity:
            # 转换severity为枚举值
            severity_level = None
            if severity.isdigit():
                severity_level = int(severity)
            elif severity.lower() == "mild":
                severity_level = SeverityLevel.MILD.value
            elif severity.lower() == "moderate":
                severity_level = SeverityLevel.MODERATE.value
            elif severity.lower() == "severe":
                severity_level = SeverityLevel.SEVERE.value
                
            templates = template_manager.get_templates_by_severity(severity_level)
        else:
            templates = template_manager.get_all_templates()
        
        # 转换为JSON可序列化的格式
        template_list = []
        for template in templates:
            template_list.append(template.save_to_json())
        
        return jsonify({
            "status": "success",
            "templates": template_list,
            "count": len(template_list),
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception("获取康复模板失败")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_template_bp.route('/<template_id>', methods=['GET'])
@api_error_handler
def get_template(template_id):
    """获取特定康复模板"""
    try:
        # 获取模板管理器
        template_manager = current_app.extensions.get('rehab_template_manager')
        if not template_manager:
            return jsonify({
                "status": "error", 
                "message": "模板管理器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 获取模板
        template = template_manager.get_template(template_id)
        if not template:
            return jsonify({
                "status": "error",
                "message": f"模板 {template_id} 不存在",
                "timestamp": datetime.now().timestamp()
            }), 404
        
        # 转换为JSON可序列化的格式
        template_data = template.save_to_json()
        
        return jsonify({
            "status": "success",
            "template": template_data,
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception(f"获取康复模板 {template_id} 失败")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_template_bp.route('/create', methods=['POST'])
@api_error_handler
@rate_limiter(limit=5, period=60)  # 限制每分钟最多5次请求
def create_template():
    """创建新的康复模板"""
    try:
        # 获取模板管理器
        template_manager = current_app.extensions.get('rehab_template_manager')
        if not template_manager:
            return jsonify({
                "status": "error", 
                "message": "模板管理器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 获取请求数据
        template_data = request.get_json()
        if not template_data:
            return jsonify({
                "status": "error",
                "message": "请求中未包含模板数据",
                "timestamp": datetime.now().timestamp()
            }), 400
        
        # 验证必要字段
        required_fields = ['template_id', 'name', 'template_type', 'severity_level']
        for field in required_fields:
            if field not in template_data:
                return jsonify({
                    "status": "error",
                    "message": f"缺少必要字段: {field}",
                    "timestamp": datetime.now().timestamp()
                }), 400
        
        # 检查模板ID是否已存在
        existing_template = template_manager.get_template(template_data['template_id'])
        if existing_template:
            return jsonify({
                "status": "error",
                "message": f"模板ID '{template_data['template_id']}' 已存在",
                "timestamp": datetime.now().timestamp()
            }), 400
        
        # 创建模板
        template = template_manager.create_template(template_data)
        if not template:
            return jsonify({
                "status": "error",
                "message": "创建模板失败",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 返回创建的模板
        return jsonify({
            "status": "success",
            "message": "模板创建成功",
            "template": template.save_to_json(),
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception("创建康复模板失败")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_template_bp.route('/update/<template_id>', methods=['PUT'])
@api_error_handler
@rate_limiter(limit=5, period=60)  # 限制每分钟最多5次请求
def update_template(template_id):
    """更新康复模板"""
    try:
        # 获取模板管理器
        template_manager = current_app.extensions.get('rehab_template_manager')
        if not template_manager:
            return jsonify({
                "status": "error", 
                "message": "模板管理器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 检查模板是否存在
        existing_template = template_manager.get_template(template_id)
        if not existing_template:
            return jsonify({
                "status": "error",
                "message": f"模板 {template_id} 不存在",
                "timestamp": datetime.now().timestamp()
            }), 404
        
        # 获取请求数据
        template_data = request.get_json()
        if not template_data:
            return jsonify({
                "status": "error",
                "message": "请求中未包含模板数据",
                "timestamp": datetime.now().timestamp()
            }), 400
        
        # 更新模板
        success = template_manager.update_template(template_id, template_data)
        if not success:
            return jsonify({
                "status": "error",
                "message": "更新模板失败",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 获取更新后的模板
        updated_template = template_manager.get_template(template_id)
        
        # 返回更新后的模板
        return jsonify({
            "status": "success",
            "message": "模板更新成功",
            "template": updated_template.save_to_json(),
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception(f"更新康复模板 {template_id} 失败")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_template_bp.route('/delete/<template_id>', methods=['DELETE'])
@api_error_handler
@rate_limiter(limit=3, period=60)  # 限制每分钟最多3次请求
def delete_template(template_id):
    """删除康复模板"""
    try:
        # 获取模板管理器
        template_manager = current_app.extensions.get('rehab_template_manager')
        if not template_manager:
            return jsonify({
                "status": "error", 
                "message": "模板管理器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 检查模板是否存在
        existing_template = template_manager.get_template(template_id)
        if not existing_template:
            return jsonify({
                "status": "error",
                "message": f"模板 {template_id} 不存在",
                "timestamp": datetime.now().timestamp()
            }), 404
        
        # 删除模板
        success = template_manager.delete_template(template_id)
        if not success:
            return jsonify({
                "status": "error",
                "message": "删除模板失败",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 返回成功响应
        return jsonify({
            "status": "success",
            "message": f"模板 {template_id} 已成功删除",
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception(f"删除康复模板 {template_id} 失败")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_template_bp.route('/match', methods=['POST'])
@api_error_handler
def match_template():
    """将检测到的姿势与模板进行匹配"""
    try:
        # 获取模板管理器
        template_manager = current_app.extensions.get('rehab_template_manager')
        if not template_manager:
            return jsonify({
                "status": "error", 
                "message": "模板管理器未初始化",
                "timestamp": datetime.now().timestamp()
            }), 500
        
        # 获取请求数据
        data = request.get_json()
        if not data or 'keypoints' not in data or 'template_id' not in data:
            return jsonify({
                "status": "error",
                "message": "请求中未包含必要数据（关键点和模板ID）",
                "timestamp": datetime.now().timestamp()
            }), 400
        
        keypoints = data['keypoints']
        template_id = data['template_id']
        frame_size = data.get('frame_size')  # 可选参数
        
        # 获取模板
        template = template_manager.get_template(template_id)
        if not template:
            return jsonify({
                "status": "error",
                "message": f"模板 {template_id} 不存在",
                "timestamp": datetime.now().timestamp()
            }), 404
        
        # 匹配姿势
        score, feedback = template.match_pose(keypoints, frame_size)
        
        # 返回匹配结果
        return jsonify({
            "status": "success",
            "match_result": {
                "score": score,
                "feedback": feedback,
                "template_id": template_id,
                "template_name": template.name
            },
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception("姿势匹配失败")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_template_bp.route('/severity-types', methods=['GET'])
@api_error_handler
def get_severity_types():
    """获取严重程度类型列表"""
    try:
        severities = [
            {"id": SeverityLevel.MILD.value, "name": "轻度", "description": "Cobb角10-25度"},
            {"id": SeverityLevel.MODERATE.value, "name": "中度", "description": "Cobb角25-40度"},
            {"id": SeverityLevel.SEVERE.value, "name": "重度", "description": "Cobb角>40度"}
        ]
        
        return jsonify({
            "status": "success",
            "severities": severities,
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception("获取严重程度类型失败")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

@rehab_template_bp.route('/template-types', methods=['GET'])
@api_error_handler
def get_template_types():
    """获取模板类型列表"""
    try:
        template_types = [
            {"id": RehabTemplateType.SPINE_STRETCH.value, "name": "脊柱伸展", "description": "以伸展脊柱为主的康复动作"},
            {"id": RehabTemplateType.SIDE_BEND.value, "name": "侧弯拉伸", "description": "以侧弯拉伸为主的康复动作"},
            {"id": RehabTemplateType.ROTATION.value, "name": "旋转动作", "description": "以旋转为主的康复动作"},
            {"id": RehabTemplateType.LYING.value, "name": "仰卧/侧卧动作", "description": "卧姿康复动作"},
            {"id": RehabTemplateType.CUSTOM.value, "name": "自定义动作", "description": "用户自定义的康复动作"}
        ]
        
        return jsonify({
            "status": "success",
            "template_types": template_types,
            "timestamp": datetime.now().timestamp()
        })
        
    except Exception as e:
        current_app.logger.exception("获取模板类型失败")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().timestamp()
        }), 500

def register_rehab_template_routes(app):
    """注册康复模板相关路由"""
    # 注册蓝图
    app.register_blueprint(rehab_template_bp)
    
    # 初始化模板管理器
    from app.models.rehab.rehab_template import RehabTemplateManager
    template_manager = RehabTemplateManager()
    
    # 将模板管理器存储到应用扩展中
    app.extensions['rehab_template_manager'] = template_manager
    
    app.logger.info("康复模板路由已注册") 