# 移除循环导入
# from app.routes.rehabilitation import register_rehabilitation_routes
# from app.routes.video_stream import register_video_routes

def init_rehabilitation_module(app):
    """初始化康复指导模块"""
    # 在这里不直接导入，而是在函数内部导入
    from app.routes.rehabilitation import register_rehabilitation_routes
    from app.routes.video_stream import register_video_routes
    
    register_rehabilitation_routes(app)
    register_video_routes(app)
    
    print("脊柱康复指导模块已初始化") 