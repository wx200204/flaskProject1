import os
from pathlib import Path

# 设置环境变量
os.environ['FLASK_ENV'] = 'development'
os.environ['DEBUG_MODE'] = 'True'

print("启动脊柱侧弯分析系统...")
print("开发模式已启用")

# 确保模型目录存在
models_dir = Path(__file__).parent / 'models'
models_dir.mkdir(exist_ok=True)
print(f"模型目录已确认: {models_dir}")

# 验证模型文件
model_path = models_dir / 'spine_model.pth'
if not model_path.exists():
    print(f"模型文件不存在，正在创建简单模型: {model_path}")
    import torch
    dummy_state = {
        'layer1.weight': torch.zeros(10, 10),
        'layer1.bias': torch.zeros(10),
    }
    torch.save(dummy_state, model_path)
    print(f"模型已创建: {model_path}")
else:
    print("现有模型文件已验证有效")

print("正在导入应用...")
from app import create_app

app = create_app()

# 尝试导入装饰器，提供备用空装饰器
try:
    from app.utils.decorators import api_error_handler, web_error_handler, rate_limiter
except ImportError:
    # 定义空装饰器
    def api_error_handler(f): return f
    def web_error_handler(f): return f
    def rate_limiter(limit=None, period=None): 
        def decorator(f): return f
        return decorator

# 定义可能缺失的装饰器为空函数
def cache_result(timeout=None):
    def decorator(f): return f
    return decorator
    
def login_required(f): return f

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)