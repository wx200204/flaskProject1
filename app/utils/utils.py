# app/utils/utils.py
import os
import json
from flask import current_app
import cv2
import time
import uuid
from datetime import datetime

def save_report(original_image, result_image, cobb_angle=0.0, confidence=0.0, severity="未知"):
    """存储分析报告和相关图像
    
    Args:
        original_image: 原始图像
        result_image: 分析结果图像
        cobb_angle: 计算的Cobb角度
        confidence: 结果置信度
        severity: 严重程度
        
    Returns:
        报告文件路径
    """
    # 生成唯一报告ID
    report_id = f"report_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    # 创建报告目录
    reports_dir = current_app.config.get('REPORTS_DIR', os.path.join(current_app.root_path, 'static/reports'))
    os.makedirs(reports_dir, exist_ok=True)
    
    # 保存图像
    report_path = os.path.join(reports_dir, f"{report_id}")
    
    # 保存原始图像和结果图像
    original_path = f"{report_path}_original.jpg"
    result_path = f"{report_path}_result.jpg"
    
    cv2.imwrite(original_path, original_image)
    cv2.imwrite(result_path, result_image)
    
    # 创建报告数据
    report_data = {
        "id": report_id,
        "timestamp": datetime.now().isoformat(),
        "cobb_angle": float(cobb_angle),
        "confidence": float(confidence),
        "severity": severity,
        "original_image": f"/static/reports/{os.path.basename(original_path)}",
        "result_image": f"/static/reports/{os.path.basename(result_path)}"
    }
    
    # 保存JSON报告数据
    with open(f"{report_path}.json", 'w') as f:
        json.dump(report_data, f, indent=2)
    
    return report_path