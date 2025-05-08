import os
from flask import Flask
from flask_cors import CORS
from .config import db
from .routes import bp
from .utils.rehab import init_rehab_module
from .utils.database_service import request_logger_middleware
# 导入新的路由模块
from .routes.main import main_bp
from .routes.rehabilitation import rehab_bp, register_rehabilitation_routes
from .routes.video_stream import video_bp, register_video_routes
from .routes.api import api_bp  # 导入新的API蓝图
from .routes.posture import posture_bp as posture_eval_bp, register_posture_routes  # 导入体态评估蓝图，使用别名避免冲突
from .routes.rehab_template import rehab_template_bp, register_rehab_template_routes  # 导入康复模板蓝图

def create_app(test_config=None):
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # 启用CORS
    CORS(app)
    
    # 配置应用
    if test_config is None:
        app.config.from_mapping(
            SECRET_KEY='dev',
            UPLOAD_DIR=os.path.join(app.root_path, 'uploads'),
            MODEL_DIR=os.path.join(app.root_path, 'models'),
            LOG_DIR=os.path.join(app.root_path, 'logs'),
            MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 限制上传文件大小为16MB
            REHAB_IMAGES_DIR=os.path.join(app.root_path, 'static', 'img', 'rehab'),
            REHAB_VIDEOS_DIR=os.path.join(app.root_path, 'static', 'video', 'rehab'),
            SQLALCHEMY_DATABASE_URI='mysql+pymysql://root:200204@localhost:3306/spine_analysis',
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
            SQLALCHEMY_ECHO=True
        )
    else:
        app.config.update(test_config)
    
    # 初始化数据库
    db.init_app(app)
    
    # 注册请求记录中间件
    @app.before_request
    def before_request():
        from flask import g
        import time
        g.start_time = time.time()
        
    app.after_request(request_logger_middleware())
    
    # 确保必要的目录存在
    for directory in [app.config['UPLOAD_DIR'], 
                     app.config['MODEL_DIR'],
                     app.config['LOG_DIR'],
                     app.config['REHAB_IMAGES_DIR'],
                     app.config['REHAB_VIDEOS_DIR']]:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            app.logger.warning(f"Failed to create directory {directory}: {str(e)}")
    
    # 注册蓝图 - 修改顺序，确保api_bp先注册以处理/api/v1/analyze
    app.register_blueprint(api_bp)  # 先注册新API蓝图处理/api/v1/analyze请求
    app.register_blueprint(main_bp)
    app.register_blueprint(posture_eval_bp)  # 注册体态评估蓝图
    app.register_blueprint(bp)  # 最后注册老蓝图，避免与api_bp冲突
    
    # 创建应用上下文，在上下文中初始化康复模块
    with app.app_context():
        # 创建所有数据库表
        db.create_all()
        
        # 初始化康复模块
        rehab_manager, session_manager = init_rehab_module()
        app.rehab_manager = rehab_manager
        app.session_manager = session_manager
        
        # 使用注册函数注册其他蓝图
        register_rehabilitation_routes(app)
        register_video_routes(app)
        register_posture_routes(app)  # 注册体态评估路由
        register_rehab_template_routes(app)  # 注册康复模板路由
    
    return app
