import os
from pathlib import Path

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
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:200204@localhost:3306/spine_analysis'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = True 