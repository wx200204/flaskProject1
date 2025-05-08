from .rehab_manager import RehabManager, SeverityLevel
from .session_manager import SessionManager

# 创建全局变量，以便在应用中共享实例
rehab_manager = None
session_manager = None

def init_rehab_module():
    """初始化康复模块"""
    global rehab_manager, session_manager
    
    # 初始化康复方案管理器
    if rehab_manager is None:
        rehab_manager = RehabManager()
    
    # 初始化会话管理器
    if session_manager is None:
        session_manager = SessionManager()
    
    return rehab_manager, session_manager 