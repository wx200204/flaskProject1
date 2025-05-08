from datetime import datetime
from app.config import db

class User(db.Model):
    """用户模型"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # 关系
    analysis_records = db.relationship('AnalysisRecord', backref='user', lazy=True)
    rehab_sessions = db.relationship('RehabSession', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class AnalysisRecord(db.Model):
    """图像分析记录"""
    __tablename__ = 'analysis_records'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    image_path = db.Column(db.String(255), nullable=False)
    cobb_angle = db.Column(db.Float, nullable=True)
    analysis_results = db.Column(db.JSON, nullable=True)
    model_id = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<AnalysisRecord {self.id}>'

class RehabSession(db.Model):
    """康复训练会话"""
    __tablename__ = 'rehab_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    plan_id = db.Column(db.String(50), nullable=True)
    start_time = db.Column(db.DateTime, default=datetime.now)
    end_time = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default='active')
    
    # 关系
    exercises = db.relationship('RehabExercise', backref='session', lazy=True)
    
    def __repr__(self):
        return f'<RehabSession {self.session_id}>'

class RehabExercise(db.Model):
    """康复训练项目"""
    __tablename__ = 'rehab_exercises'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('rehab_sessions.id'), nullable=False)
    exercise_id = db.Column(db.String(50), nullable=False)
    exercise_name = db.Column(db.String(100), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.now)
    end_time = db.Column(db.DateTime, nullable=True)
    duration = db.Column(db.Integer, nullable=True)  # 以秒为单位
    completion_rate = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='active')
    
    def __repr__(self):
        return f'<RehabExercise {self.exercise_name}>'

class SystemUsage(db.Model):
    """系统使用情况"""
    __tablename__ = 'system_usage'
    
    id = db.Column(db.Integer, primary_key=True)
    request_path = db.Column(db.String(255), nullable=False)
    method = db.Column(db.String(10), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    response_time = db.Column(db.Float, nullable=True)  # 响应时间(毫秒)
    status_code = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<SystemUsage {self.request_path}>' 