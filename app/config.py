import os
from pathlib import Path
from flask_sqlalchemy import SQLAlchemy

# 创建数据库实例
db = SQLAlchemy()

class Config:
    """应用配置类"""
    # 基础目录
    BASE_DIR = Path(__file__).resolve().parent
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    DEBUG = os.environ.get('DEBUG_MODE', 'True').lower() == 'true'
    
    # 目录配置
    UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    REHAB_IMAGES_DIR = os.path.join(UPLOAD_DIR, 'rehab_images')
    REHAB_VIDEOS_DIR = os.path.join(UPLOAD_DIR, 'rehab_videos')
    
    # 上传文件配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}
    
    # 数据库配置
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URI') or 'mysql+pymysql://root:200204@localhost:3306/spine_analysis'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = True

    # 模型路径
    MODEL_PATH = os.path.join(Path(__file__).parent.parent, 'models', 'spine_model.pth')
    POSE_MODEL_PATH = {
        'prototxt': str(Path(__file__).parent.parent / 'models/pose_deploy.prototxt'),
        'caffemodel': str(Path(__file__).parent.parent / 'models/pose_iter_584000.caffemodel')
    }

    # 医学参数
    COBB_THRESHOLDS = {
        'normal': 10,
        'mild': 25,
        'moderate': 40,
        'severe': 60
    }
    
    # 日志配置
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = str(Path(__file__).parent.parent / 'logs/app.log')
    
    # API配置
    API_RATE_LIMIT = os.getenv('API_RATE_LIMIT', '100/hour')
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', 30))  # 秒
    
    # 缓存配置
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'simple')
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 300))  # 秒
    
    # 安全配置
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # 调试配置
    SAVE_DEBUG_IMAGES = os.getenv('SAVE_DEBUG_IMAGES', 'False').lower() == 'true'

    # 模型配置
    MODEL_CONFIG = {
        'REQUIRE_MODEL': not DEBUG,  # 在非调试模式下要求模型
        'CREATE_DUMMY_MODEL': DEBUG,  # 在调试模式下创建虚拟模型
        'VALIDATE_ON_STARTUP': True,      # 启动时验证模型
    }

    # 确保目录存在
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)