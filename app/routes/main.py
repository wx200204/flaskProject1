from flask import Blueprint, render_template, current_app, redirect, url_for

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """主页路由"""
    return render_template('index.html')

@main_bp.route('/upload')
def upload():
    """上传页面路由"""
    return render_template('upload.html')

@main_bp.route('/rehabilitation')
def rehabilitation():
    """康复指导页面路由"""
    return render_template('rehabilitation.html')

@main_bp.route('/rehabilitation/debug')
def rehabilitation_debug():
    """康复指导调试页面路由"""
    return render_template('rehabilitation_debug.html')

@main_bp.route('/rehab/session/<session_id>/complete')
def rehab_session_complete(session_id):
    """康复训练完成页面路由"""
    return render_template('rehab_complete.html', session_id=session_id)

@main_bp.route('/posture_evaluation')
def posture_evaluation():
    """体态评估页面路由 - 重定向到新的路由"""
    return redirect(url_for('posture_eval.posture_index'))

@main_bp.route('/user/records')
def user_records():
    """用户记录页面"""
    return render_template('user_records.html', title="用户记录")

@main_bp.route('/status')
def status():
    """状态页面"""
    return render_template('status.html') 