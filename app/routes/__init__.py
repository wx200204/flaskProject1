from flask import Blueprint

# 创建主蓝图 - 使用不同的名称，避免与routes.py中的蓝图冲突
bp = Blueprint('api_routes_init', __name__, url_prefix='')

# 导入路由模块
from . import main
from . import rehabilitation
from . import video_stream
from . import api  # 导入新的API模块 