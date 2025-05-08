import functools
import time
from flask import jsonify, request, current_app
import traceback

def api_error_handler(f):
    """API错误处理装饰器，捕获异常并返回JSON格式的错误信息"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            current_app.logger.error(f"API错误: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
    return decorated_function

def rate_limiter(limit=10, period=60):
    """请求频率限制装饰器
    
    Args:
        limit: 在指定时间段内允许的最大请求数
        period: 时间段，单位为秒
    """
    def decorator(f):
        # 使用闭包存储请求历史
        request_history = {}
        
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # 获取客户端IP
            client_ip = request.remote_addr
            
            # 获取当前时间
            current_time = time.time()
            
            # 初始化该IP的请求历史
            if client_ip not in request_history:
                request_history[client_ip] = []
            
            # 清理过期的请求记录
            request_history[client_ip] = [t for t in request_history[client_ip] 
                                         if current_time - t < period]
            
            # 检查是否超过限制
            if len(request_history[client_ip]) >= limit:
                return jsonify({
                    "status": "error",
                    "message": "请求频率过高，请稍后再试"
                }), 429
            
            # 记录本次请求
            request_history[client_ip].append(current_time)
            
            # 执行原函数
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator 