import logging
import json
import time
import uuid
import os
from datetime import datetime
from flask import current_app
import cv2
import numpy as np
from ..utils import save_report
from ...models.rehab.pose_detector import PoseDetector, PoseStatus
from .rehab_manager import RehabManager

class SessionManager:
    """康复会话管理器"""
    
    def __init__(self):
        """初始化康复会话管理器"""
        self.logger = logging.getLogger(__name__)
        self.active_sessions = {}  # 存储活动的康复会话
        self.pose_detector = PoseDetector()  # 姿势检测器
        self.rehab_manager = RehabManager()  # 康复方案管理器
    
    def create_session(self, user_id, cobb_angle, user_info=None):
        """创建新的康复会话
        
        Args:
            user_id: 用户ID
            cobb_angle: Cobb角度
            user_info: 用户信息，可选
            
        Returns:
            session_id: 会话ID
            plan: 康复方案
        """
        try:
            # 生成康复方案
            plan = self.rehab_manager.generate_rehab_plan(cobb_angle, user_info)
            
            # 创建会话ID
            session_id = str(uuid.uuid4())
            
            # 创建会话记录
            session = {
                "session_id": session_id,
                "user_id": user_id,
                "cobb_angle": cobb_angle,
                "plan": plan,
                "start_time": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "completed_exercises": [],
                "current_exercise_index": 0,
                "status": "active",
                "scores": []
            }
            
            # 保存会话
            self.active_sessions[session_id] = session
            
            return session_id, plan
            
        except Exception as e:
            self.logger.error(f"创建康复会话失败: {str(e)}")
            return None, None
    
    def get_session(self, session_id):
        """获取会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            session: 会话信息
        """
        return self.active_sessions.get(session_id)
    
    def update_session_status(self, session_id, status):
        """更新会话状态
        
        Args:
            session_id: 会话ID
            status: 新状态
            
        Returns:
            success: 是否成功
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = status
            self.active_sessions[session_id]["last_active"] = datetime.now().isoformat()
            return True
        return False
    
    def get_current_exercise(self, session_id):
        """获取当前训练项目
        
        Args:
            session_id: 会话ID
            
        Returns:
            exercise: 训练项目
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        if session["current_exercise_index"] >= len(session["plan"]["exercises"]):
            return None  # 所有训练项目已完成
        
        return session["plan"]["exercises"][session["current_exercise_index"]]
    
    def move_to_next_exercise(self, session_id):
        """移动到下一个训练项目
        
        Args:
            session_id: 会话ID
            
        Returns:
            next_exercise: 下一个训练项目，如果没有则返回None
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        # 记录当前训练完成
        current_exercise = self.get_current_exercise(session_id)
        if current_exercise:
            session["completed_exercises"].append({
                "exercise_id": current_exercise["id"],
                "completed_at": datetime.now().isoformat(),
                "score": session["scores"][-1] if session["scores"] else 0
            })
        
        # 移动到下一个训练
        session["current_exercise_index"] += 1
        session["last_active"] = datetime.now().isoformat()
        
        # 检查是否所有训练都已完成
        if session["current_exercise_index"] >= len(session["plan"]["exercises"]):
            session["status"] = "completed"
            self._save_session_report(session)
            return None
        
        return self.get_current_exercise(session_id)
    
    def evaluate_pose(self, session_id, image):
        """评估姿势
        
        Args:
            session_id: 会话ID
            image: 输入图像
            
        Returns:
            result: 评估结果
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return {"error": "会话不存在"}
            
            current_exercise = self.get_current_exercise(session_id)
            if not current_exercise:
                return {"error": "当前没有训练项目"}
            
            # 检测姿势
            landmarks, annotated_image = self.pose_detector.detect_pose(image)
            
            # 评估姿势质量
            status, score, feedback = self.pose_detector.evaluate_pose_quality(
                landmarks, current_exercise["type"])
            
            # 记录分数
            session["scores"].append(score)
            
            # 增加平均分数
            avg_score = sum(session["scores"]) / len(session["scores"]) if session["scores"] else 0
            
            # 转换图像以便返回
            _, buffer = cv2.imencode('.jpg', annotated_image)
            image_base64 = buffer.tobytes()
            
            result = {
                "session_id": session_id,
                "exercise_id": current_exercise["id"],
                "status": status.name,
                "score": score,
                "avg_score": avg_score,
                "feedback": feedback,
                "annotated_image": image_base64
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"评估姿势失败: {str(e)}")
            return {"error": f"评估失败: {str(e)}"}
    
    def end_session(self, session_id):
        """结束会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            success: 是否成功
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session["status"] = "completed"
        session["end_time"] = datetime.now().isoformat()
        
        # 生成并保存会话报告
        self._save_session_report(session)
        
        # 移除活动会话
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        return True
    
    def _save_session_report(self, session):
        """保存会话报告
        
        Args:
            session: 会话信息
        """
        try:
            # 计算总分
            total_score = sum(session["scores"]) / len(session["scores"]) if session["scores"] else 0
            
            # 创建报告数据
            report_data = {
                "user_id": session["user_id"],
                "session_id": session["session_id"],
                "cobb_angle": session["cobb_angle"],
                "start_time": session["start_time"],
                "end_time": session.get("end_time", datetime.now().isoformat()),
                "total_score": total_score,
                "severity": session["plan"]["severity"],
                "completed_exercises": session["completed_exercises"],
                "recommendations": session["plan"]["recommendation"]
            }
            
            # 保存报告
            report_path = os.path.join(current_app.config["LOG_DIR"], "rehab_reports")
            os.makedirs(report_path, exist_ok=True)
            
            filename = f"rehab_report_{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = os.path.join(report_path, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"康复会话报告已保存: {file_path}")
            
        except Exception as e:
            self.logger.error(f"保存康复会话报告失败: {str(e)}")
    
    def clean_inactive_sessions(self, max_inactive_minutes=60):
        """清理不活跃的会话
        
        Args:
            max_inactive_minutes: 最大不活跃时间，单位为分钟
            
        Returns:
            cleaned: 清理的会话数量
        """
        try:
            now = datetime.now()
            sessions_to_remove = []
            
            for session_id, session in self.active_sessions.items():
                last_active = datetime.fromisoformat(session["last_active"])
                inactive_minutes = (now - last_active).total_seconds() / 60
                
                if inactive_minutes > max_inactive_minutes:
                    sessions_to_remove.append(session_id)
            
            # 清理会话
            for session_id in sessions_to_remove:
                session = self.active_sessions[session_id]
                session["status"] = "timeout"
                session["end_time"] = now.isoformat()
                self._save_session_report(session)
                del self.active_sessions[session_id]
            
            return len(sessions_to_remove)
            
        except Exception as e:
            self.logger.error(f"清理不活跃会话失败: {str(e)}")
            return 0 