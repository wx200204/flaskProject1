from datetime import datetime
from flask import current_app, request, g
from app.config import db
from app.models.database import User, AnalysisRecord, RehabSession, RehabExercise, SystemUsage
import time
import uuid

class DatabaseService:
    """数据库服务类，处理与数据库相关的操作"""
    
    @staticmethod
    def get_or_create_user(username, email=None):
        """获取或创建用户"""
        user = User.query.filter_by(username=username).first()
        if not user:
            user = User(username=username, email=email)
            db.session.add(user)
            db.session.commit()
        return user
    
    @staticmethod
    def record_analysis(user_id=None, image_path=None, cobb_angle=None, results=None, model_id=None):
        """记录图像分析结果"""
        record = AnalysisRecord(
            user_id=user_id,
            image_path=image_path,
            cobb_angle=cobb_angle,
            analysis_results=results,
            model_id=model_id
        )
        db.session.add(record)
        db.session.commit()
        return record
    
    @staticmethod
    def create_rehab_session(user_id=None, plan_id=None):
        """创建康复训练会话"""
        session_id = str(uuid.uuid4())
        session = RehabSession(
            session_id=session_id,
            user_id=user_id,
            plan_id=plan_id,
            status='active'
        )
        db.session.add(session)
        db.session.commit()
        return session
    
    @staticmethod
    def record_rehab_exercise(session_id, exercise_id, exercise_name):
        """记录康复训练项目"""
        # 获取会话对象
        session = RehabSession.query.filter_by(session_id=session_id).first()
        if not session:
            return None
            
        exercise = RehabExercise(
            session_id=session.id,
            exercise_id=exercise_id,
            exercise_name=exercise_name,
            status='active'
        )
        db.session.add(exercise)
        db.session.commit()
        return exercise
    
    @staticmethod
    def complete_rehab_exercise(session_id, exercise_id, completion_rate=100.0):
        """完成康复训练项目"""
        # 获取会话对象
        session = RehabSession.query.filter_by(session_id=session_id).first()
        if not session:
            return None
            
        # 获取训练项目
        exercise = RehabExercise.query.filter_by(
            session_id=session.id,
            exercise_id=exercise_id,
            status='active'
        ).first()
        
        if exercise:
            end_time = datetime.now()
            duration = (end_time - exercise.start_time).total_seconds()
            
            exercise.end_time = end_time
            exercise.duration = int(duration)
            exercise.completion_rate = completion_rate
            exercise.status = 'completed'
            
            db.session.commit()
            return exercise
        return None
    
    @staticmethod
    def complete_rehab_session(session_id):
        """完成康复训练会话"""
        session = RehabSession.query.filter_by(session_id=session_id).first()
        if session:
            session.end_time = datetime.now()
            session.status = 'completed'
            db.session.commit()
            return session
        return None
    
    @staticmethod
    def record_system_usage(request_path=None, method=None, user_id=None, response_time=None, status_code=None):
        """记录系统使用情况"""
        usage = SystemUsage(
            request_path=request_path or request.path,
            method=method or request.method,
            user_id=user_id,
            response_time=response_time,
            status_code=status_code
        )
        db.session.add(usage)
        db.session.commit()
        return usage
    
    @staticmethod
    def get_user_analysis_records(user_id, limit=10):
        """获取用户的分析记录"""
        return AnalysisRecord.query.filter_by(user_id=user_id).order_by(AnalysisRecord.created_at.desc()).limit(limit).all()
    
    @staticmethod
    def get_user_rehab_sessions(user_id, limit=10):
        """获取用户的康复训练会话"""
        return RehabSession.query.filter_by(user_id=user_id).order_by(RehabSession.start_time.desc()).limit(limit).all()
    
    @staticmethod
    def get_session_exercises(session_id):
        """获取会话中的训练项目"""
        session = RehabSession.query.filter_by(session_id=session_id).first()
        if not session:
            return []
        return RehabExercise.query.filter_by(session_id=session.id).order_by(RehabExercise.start_time).all()


# 创建请求记录中间件
def request_logger_middleware():
    """请求记录中间件，用于记录请求响应时间和状态"""
    def after_request(response):
        # 在请求处理时记录开始时间，而不是中间件初始化时
        response_time = None
        if hasattr(g, 'start_time'):
            response_time = (time.time() - g.start_time) * 1000  # 毫秒
        user_id = getattr(g, 'user_id', None)
        
        # 异步记录请求信息
        try:
            if response_time is not None:
                DatabaseService.record_system_usage(
                    request_path=request.path,
                    method=request.method,
                    user_id=user_id,
                    response_time=response_time,
                    status_code=response.status_code
                )
        except Exception as e:
            # 记录错误但不影响响应
            print(f"记录请求信息失败: {str(e)}")
            
        return response
    
    return after_request 