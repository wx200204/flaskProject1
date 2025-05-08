import os
import json
import csv
import zipfile
from flask import current_app
from datetime import datetime


class DataExporter:
    """数据导出工具"""
    
    def __init__(self, reports_dir=None):
        if reports_dir is None:
            self.reports_dir = current_app.config['REPORTS_DIR']
        else:
            self.reports_dir = reports_dir
            
    def export_to_csv(self, output_path=None):
        """Export reports to CSV using built-in csv module"""
        if output_path is None:
            output_path = os.path.join(current_app.config['UPLOAD_DIR'], 
                                      f"spine_reports_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
            
        # Read all reports
        reports_data = []
        for filename in os.listdir(self.reports_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.reports_dir, filename), 'r') as f:
                        data = json.load(f)
                        
                    # Extract report ID and time
                    report_id = filename.split('.')[0]
                    timestamp = report_id.split('-')[1] if len(report_id.split('-')) > 1 else ""
                    date = datetime.strptime(timestamp, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S') if timestamp else ""
                    
                    # Extract key data
                    report_row = {
                        "report_id": report_id,
                        "date": date,
                        "cobb_angle": data.get("cobb_angle", ""),
                        "severity": data.get("severity", ""),
                        "status": data.get("status", "")
                    }
                    
                    reports_data.append(report_row)
                except Exception as e:
                    current_app.logger.warning(f"Failed to process report {filename}: {str(e)}")
                    
        # If there's no data, return None
        if not reports_data:
            return None
            
        # Write to CSV using csv module
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ["report_id", "date", "cobb_angle", "severity", "status"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in reports_data:
                writer.writerow(row)
        
        return output_path
        
    def export_to_excel(self, output_path=None):
        """将所有报告导出为CSV文件 (Excel替代方案)"""
        # 由于pandas安装问题，我们使用CSV作为Excel的替代方案
        if output_path is None:
            output_path = os.path.join(current_app.config['UPLOAD_DIR'], 
                                      f"spine_reports_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
        
        # 使用CSV导出功能
        return self.export_to_csv(output_path)
        
    def export_all_data(self, output_path=None):
        """导出所有数据（报告、图像等）为ZIP文件"""
        if output_path is None:
            output_path = os.path.join(current_app.config['UPLOAD_DIR'], 
                                      f"spine_data_export_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip")
            
        # 创建ZIP文件
        with zipfile.ZipFile(output_path, 'w') as zipf:
            # 添加报告文件
            for filename in os.listdir(self.reports_dir):
                if filename.endswith('.json'):
                    zipf.write(os.path.join(self.reports_dir, filename), 
                              os.path.join('reports', filename))
                    
            # 添加可视化图像
            vis_dir = os.path.join(current_app.config['UPLOAD_DIR'], 'visualizations')
            if os.path.exists(vis_dir):
                for filename in os.listdir(vis_dir):
                    zipf.write(os.path.join(vis_dir, filename), 
                              os.path.join('visualizations', filename))
                    
            # 添加CSV导出
            csv_path = self.export_to_csv()
            if csv_path:
                zipf.write(csv_path, os.path.basename(csv_path))
                os.remove(csv_path)  # 删除临时CSV文件
                
        return output_path 