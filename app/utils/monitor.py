import os
import psutil
import platform
import torch
import time
from flask import current_app


class SystemMonitor:
    """系统监控工具"""
    
    def __init__(self):
        self.start_time = time.time()
        
    def get_system_info(self):
        """获取系统信息"""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "disk_total": psutil.disk_usage('/').total,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
        }
        
    def get_resource_usage(self):
        """获取资源使用情况"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "uptime": time.time() - self.start_time
        }
        
    def get_app_stats(self):
        """获取应用统计信息"""
        # 统计已分析的报告数量
        reports_dir = current_app.config['REPORTS_DIR']
        report_count = len([f for f in os.listdir(reports_dir) if f.endswith('.json')])
        
        # 统计上传的图像数量
        uploads_dir = current_app.config['UPLOAD_DIR']
        image_count = len([f for f in os.listdir(uploads_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # 获取最近的分析时间
        latest_report_time = None
        if report_count > 0:
            report_files = [f for f in os.listdir(reports_dir) if f.endswith('.json')]
            if report_files:
                latest_file = max(report_files, key=lambda f: os.path.getmtime(os.path.join(reports_dir, f)))
                latest_report_time = os.path.getmtime(os.path.join(reports_dir, latest_file))
        
        return {
            "total_reports": report_count,
            "total_images": image_count,
            "latest_analysis_time": latest_report_time
        }
        
    def check_health(self):
        """系统健康检查"""
        health_status = {
            "status": "healthy",
            "checks": {}
        }
        
        # 检查CPU使用率
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > 90:
            health_status["checks"]["cpu"] = {
                "status": "warning",
                "message": f"High CPU usage: {cpu_percent}%"
            }
            health_status["status"] = "warning"
        else:
            health_status["checks"]["cpu"] = {
                "status": "healthy",
                "message": f"CPU usage: {cpu_percent}%"
            }
            
        # 检查内存使用率
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            health_status["checks"]["memory"] = {
                "status": "warning",
                "message": f"High memory usage: {memory_percent}%"
            }
            health_status["status"] = "warning"
        else:
            health_status["checks"]["memory"] = {
                "status": "healthy",
                "message": f"Memory usage: {memory_percent}%"
            }
            
        # 检查磁盘使用率
        disk_percent = psutil.disk_usage('/').percent
        if disk_percent > 90:
            health_status["checks"]["disk"] = {
                "status": "warning",
                "message": f"High disk usage: {disk_percent}%"
            }
            health_status["status"] = "warning"
        else:
            health_status["checks"]["disk"] = {
                "status": "healthy",
                "message": f"Disk usage: {disk_percent}%"
            }
            
        # 检查模型可用性
        try:
            model_path = current_app.config['MODEL_PATH']
            if os.path.exists(model_path):
                health_status["checks"]["model"] = {
                    "status": "healthy",
                    "message": "Model file is available"
                }
            else:
                health_status["checks"]["model"] = {
                    "status": "critical",
                    "message": "Model file is missing"
                }
                health_status["status"] = "critical"
        except Exception as e:
            health_status["checks"]["model"] = {
                "status": "critical",
                "message": f"Model check failed: {str(e)}"
            }
            health_status["status"] = "critical"
            
        return health_status 