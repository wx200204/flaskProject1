import cv2
import numpy as np
import time
from flask import current_app
import os
from pathlib import Path


class PoseMatcher:
    """康复动作匹配评估器 - 实时比对用户动作与标准动作轮廓"""
    
    def __init__(self):
        """初始化动作匹配器"""
        self.pose_models = {}
        self.feedback_thresholds = {
            'excellent': 85,
            'good': 70,
            'moderate': 60,
            'poor': 45
        }
        self.last_feedback_time = 0
        self.feedback_cooldown = 1.0  # 反馈冷却时间（秒）
        self.consecutive_correct = 0  # 连续正确次数
        self.required_correct = 5     # 所需连续正确次数
        
        # 加载动作模板
        self._load_pose_templates()
    
    def _load_pose_templates(self):
        """加载预定义的康复动作模板"""
        try:
            # 从应用配置中获取模板目录
            if current_app:
                templates_dir = Path(current_app.config.get('REHAB_IMAGES_DIR'))
            else:
                templates_dir = Path(__file__).parent.parent.parent / 'app/static/img/rehab'
            
            # 确保目录存在
            os.makedirs(templates_dir, exist_ok=True)
            
            # 定义基本康复动作
            poses = [
                'spine_stretch',      # 脊柱伸展
                'lateral_bend',       # 侧弯矫正
                'pelvic_tilt',        # 骨盆倾斜
                'shoulder_roll',      # 肩部滚动
                'cat_cow_stretch'     # 猫牛式伸展
            ]
            
            # 为每个动作加载模板
            for pose in poses:
                template_path = templates_dir / f"{pose}_template.png"
                keypoints_path = templates_dir / f"{pose}_keypoints.npy"
                
                # 如果模板文件不存在，生成默认模板
                if not template_path.exists():
                    self._generate_default_template(pose, template_path, keypoints_path)
                
                # 加载模板图像
                if template_path.exists():
                    template = cv2.imread(str(template_path))
                    
                    # 加载关键点数据（如果存在）
                    keypoints = None
                    if keypoints_path.exists():
                        try:
                            keypoints = np.load(str(keypoints_path), allow_pickle=True)
                        except Exception as e:
                            if current_app:
                                current_app.logger.warning(f"无法加载关键点数据: {e}")
                    
                    # 存储模板数据
                    self.pose_models[pose] = {
                        'template': template,
                        'keypoints': keypoints,
                        'regions': self._extract_pose_regions(template)
                    }
                    
                    if current_app:
                        current_app.logger.debug(f"已加载康复动作模板: {pose}")
                else:
                    if current_app:
                        current_app.logger.warning(f"找不到康复动作模板: {pose}")
        
        except Exception as e:
            if current_app:
                current_app.logger.error(f"加载康复动作模板失败: {str(e)}")
    
    def _generate_default_template(self, pose_name, template_path, keypoints_path):
        """生成默认动作模板及其关键点"""
        try:
            # 创建模板目录
            os.makedirs(template_path.parent, exist_ok=True)
            
            # 创建空白画布
            template = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 根据动作类型绘制不同的指导轮廓
            if pose_name == 'spine_stretch':
                # 绘制脊柱伸展指导
                # 人体中心轮廓
                center_x, center_y = 320, 240
                # 绘制人体轮廓
                cv2.ellipse(template, (center_x, center_y-80), (50, 70), 0, 0, 360, (0, 120, 255), 2)  # 头部
                cv2.line(template, (center_x, center_y-10), (center_x, center_y+100), (0, 120, 255), 2)  # 脊柱
                cv2.line(template, (center_x, center_y+30), (center_x-70, center_y-30), (0, 120, 255), 2)  # 左臂
                cv2.line(template, (center_x, center_y+30), (center_x+70, center_y-30), (0, 120, 255), 2)  # 右臂
                cv2.line(template, (center_x, center_y+100), (center_x-50, center_y+200), (0, 120, 255), 2)  # 左腿
                cv2.line(template, (center_x, center_y+100), (center_x+50, center_y+200), (0, 120, 255), 2)  # 右腿
                
                # 生成关键点数据 (x, y, confidence)
                keypoints = np.array([
                    [center_x, center_y-80, 1.0],  # 头部
                    [center_x, center_y-10, 1.0],  # 颈部
                    [center_x, center_y+30, 1.0],  # 肩部
                    [center_x-70, center_y-30, 1.0],  # 左手
                    [center_x+70, center_y-30, 1.0],  # 右手
                    [center_x, center_y+100, 1.0],  # 髋部
                    [center_x-50, center_y+200, 1.0],  # 左脚
                    [center_x+50, center_y+200, 1.0]   # 右脚
                ])
                
            elif pose_name == 'lateral_bend':
                # 侧弯矫正动作
                center_x, center_y = 320, 240
                # 绘制S形脊柱矫正轮廓
                pts = np.array([
                    [center_x-20, center_y-100],
                    [center_x+10, center_y-50],
                    [center_x-10, center_y],
                    [center_x+10, center_y+50],
                    [center_x, center_y+100]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(template, [pts], False, (0, 255, 120), 3)
                # 添加身体其他部分
                cv2.ellipse(template, (center_x-20, center_y-130), (40, 50), 0, 0, 360, (0, 255, 120), 2)  # 头部
                cv2.line(template, (center_x, center_y), (center_x-80, center_y-20), (0, 255, 120), 2)  # 左臂
                cv2.line(template, (center_x, center_y), (center_x+80, center_y-20), (0, 255, 120), 2)  # 右臂
                cv2.line(template, (center_x, center_y+100), (center_x-40, center_y+200), (0, 255, 120), 2)  # 左腿
                cv2.line(template, (center_x, center_y+100), (center_x+40, center_y+200), (0, 255, 120), 2)  # 右腿
                
                # 关键点
                keypoints = np.array([
                    [center_x-20, center_y-130, 1.0],  # 头部
                    [center_x-20, center_y-100, 1.0],  # 颈部
                    [center_x+10, center_y-50, 1.0],   # 上背部
                    [center_x-10, center_y, 1.0],      # 中背部
                    [center_x+10, center_y+50, 1.0],   # 下背部
                    [center_x, center_y+100, 1.0],     # 髋部
                    [center_x-80, center_y-20, 1.0],   # 左手
                    [center_x+80, center_y-20, 1.0],   # 右手
                    [center_x-40, center_y+200, 1.0],  # 左脚
                    [center_x+40, center_y+200, 1.0]   # 右脚
                ])
                
            else:
                # 其他动作的默认模板
                cv2.putText(template, f"{pose_name} - 标准姿势", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 简单的人形轮廓
                center_x, center_y = 320, 240
                # 头部和身体
                cv2.circle(template, (center_x, center_y-100), 40, (0, 200, 200), 2)
                cv2.line(template, (center_x, center_y-60), (center_x, center_y+60), (0, 200, 200), 2)
                # 四肢
                cv2.line(template, (center_x, center_y-20), (center_x-90, center_y), (0, 200, 200), 2)
                cv2.line(template, (center_x, center_y-20), (center_x+90, center_y), (0, 200, 200), 2)
                cv2.line(template, (center_x, center_y+60), (center_x-45, center_y+180), (0, 200, 200), 2)
                cv2.line(template, (center_x, center_y+60), (center_x+45, center_y+180), (0, 200, 200), 2)
                
                # 简单关键点
                keypoints = np.array([
                    [center_x, center_y-100, 1.0],    # 头部
                    [center_x, center_y-60, 1.0],     # 颈部
                    [center_x, center_y-20, 1.0],     # 肩部
                    [center_x-90, center_y, 1.0],     # 左手
                    [center_x+90, center_y, 1.0],     # 右手
                    [center_x, center_y+60, 1.0],     # 髋部
                    [center_x-45, center_y+180, 1.0], # 左脚
                    [center_x+45, center_y+180, 1.0]  # 右脚
                ])
            
            # 添加指导文字
            cv2.putText(template, f"请匹配姿势轮廓", (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 保存模板和关键点
            cv2.imwrite(str(template_path), template)
            np.save(str(keypoints_path), keypoints)
            
            if current_app:
                current_app.logger.info(f"已创建默认康复动作模板: {pose_name}")
                
        except Exception as e:
            if current_app:
                current_app.logger.error(f"创建默认模板失败: {str(e)}")
    
    def _extract_pose_regions(self, template):
        """从模板图像提取关键区域"""
        regions = {}
        
        if template is None:
            return regions
            
        try:
            # 转换为灰度并二值化
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 提取主要区域
            if contours:
                # 按面积排序，取最大的几个
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
                
                for i, contour in enumerate(contours):
                    # 创建掩码
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    
                    # 计算包围盒
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    regions[f'region_{i}'] = {
                        'mask': mask,
                        'contour': contour,
                        'bbox': (x, y, w, h)
                    }
            
            # 也添加预定义的身体部位区域
            h, w = template.shape[:2]
            # 头部区域 (上20%)
            regions['head'] = {
                'bbox': (0, 0, w, int(h * 0.2)),
                'weight': 0.15
            }
            # 躯干区域 (中间50%)
            regions['torso'] = {
                'bbox': (0, int(h * 0.2), w, int(h * 0.5)),
                'weight': 0.5
            }
            # 下肢区域 (下30%)
            regions['legs'] = {
                'bbox': (0, int(h * 0.7), w, int(h * 0.3)),
                'weight': 0.35
            }
            
        except Exception as e:
            if current_app:
                current_app.logger.warning(f"提取姿势区域失败: {str(e)}")
                
        return regions
    
    def evaluate_pose(self, keypoints, pose_type, image=None, low_quality=False):
        """评估用户姿势与标准姿势的匹配程度
        
        Args:
            keypoints: 用户姿势关键点 [(x,y,c), ...]
            pose_type: 动作类型 (如 'spine_stretch')
            image: 用户图像 (可选，用于可视化)
            low_quality: 是否使用低质量模式 (减少计算)
            
        Returns:
            dict: 评估结果，包含状态和分数
        """
        start_time = time.time()
        
        # 默认结果
        result = {
            'status': 'INCORRECT',
            'score': 50,
            'feedback': '请调整姿势以匹配指导轮廓',
            'details': {}
        }
        
        # 检查是否有该动作的模板
        if pose_type not in self.pose_models:
            if current_app:
                current_app.logger.warning(f"找不到姿势模板: {pose_type}")
            result['feedback'] = f"未定义的动作类型: {pose_type}"
            return result
        
        # 获取标准姿势模板
        pose_model = self.pose_models[pose_type]
        template_keypoints = pose_model.get('keypoints')
        
        if template_keypoints is None or len(template_keypoints) < 3:
            result['feedback'] = "模板关键点不足，无法评估"
            return result
        
        # 预处理关键点，确保格式一致且有效
        if isinstance(keypoints, list):
            keypoints = np.array(keypoints)
        
        # 快速质量检查 - 至少需要4个有效关键点
        valid_points = keypoints[keypoints[:, 2] > 0.2]
        if len(valid_points) < 4:
            result['feedback'] = "检测到的关键点不足，请确保全身在画面中"
            result['score'] = 30
            return result
        
        # 归一化关键点坐标 (相对于图像大小)
        if image is not None:
            img_h, img_w = image.shape[:2]
        else:
            # 使用默认尺寸
            img_h, img_w = 480, 640
            
        norm_user = keypoints.copy()
        norm_user[:, 0] = norm_user[:, 0] / img_w
        norm_user[:, 1] = norm_user[:, 1] / img_h
        
        norm_template = template_keypoints.copy()
        # 假设模板关键点已经归一化，如果没有则进行归一化
        if np.max(norm_template[:, 0]) > 1.0 or np.max(norm_template[:, 1]) > 1.0:
            template_img = pose_model.get('template')
            if template_img is not None:
                tmpl_h, tmpl_w = template_img.shape[:2]
                norm_template[:, 0] = norm_template[:, 0] / tmpl_w
                norm_template[:, 1] = norm_template[:, 1] / tmpl_h
        
        # 快速模式评估 - 减少计算量
        if low_quality:
            # 简单的欧氏距离计算
            dist_matrix = np.zeros((len(norm_user), len(norm_template)))
            for i, u_pt in enumerate(norm_user):
                for j, t_pt in enumerate(norm_template):
                    # 只考虑置信度高的点
                    if u_pt[2] > 0.2:
                        dist = np.sqrt((u_pt[0] - t_pt[0])**2 + (u_pt[1] - t_pt[1])**2)
                        dist_matrix[i, j] = dist
            
            # 计算平均距离作为相似度指标
            valid_dists = dist_matrix[dist_matrix > 0]
            if len(valid_dists) > 0:
                avg_dist = np.mean(valid_dists)
                # 距离越小，分数越高
                score = max(0, min(100, 100 - avg_dist * 200))
            else:
                score = 40
        else:
            # 详细评估 - 考虑更多因素
            # 1. 关键点匹配
            point_scores = []
            min_points = min(len(norm_user), len(norm_template))
            
            for i in range(min_points):
                if i < len(norm_user) and i < len(norm_template) and norm_user[i, 2] > 0.2:
                    # 计算归一化距离
                    dist = np.sqrt((norm_user[i, 0] - norm_template[i, 0])**2 + 
                                   (norm_user[i, 1] - norm_template[i, 1])**2)
                    
                    # 转换为分数 (距离小于0.1认为是良好匹配)
                    pt_score = max(0, min(100, 100 - dist * 300))
                    # 使用置信度加权
                    pt_score *= norm_user[i, 2]
                    point_scores.append(pt_score)
            
            # 2. 姿势形状相似度 (如果点足够多)
            shape_score = 0
            if len(point_scores) >= 5:
                # 计算标准化的姿势向量
                user_vectors = []
                template_vectors = []
                
                for i in range(min_points-1):
                    if (i < len(norm_user)-1 and i < len(norm_template)-1 and 
                        norm_user[i, 2] > 0.2 and norm_user[i+1, 2] > 0.2):
                        # 计算向量
                        u_vec = np.array([norm_user[i+1, 0] - norm_user[i, 0], 
                                         norm_user[i+1, 1] - norm_user[i, 1]])
                        t_vec = np.array([norm_template[i+1, 0] - norm_template[i, 0], 
                                         norm_template[i+1, 1] - norm_template[i, 1]])
                        
                        # 归一化向量
                        u_len = np.linalg.norm(u_vec)
                        t_len = np.linalg.norm(t_vec)
                        if u_len > 0 and t_len > 0:
                            u_vec = u_vec / u_len
                            t_vec = t_vec / t_len
                            
                            user_vectors.append(u_vec)
                            template_vectors.append(t_vec)
                
                # 计算向量相似度 (点积)
                vec_scores = []
                for u_vec, t_vec in zip(user_vectors, template_vectors):
                    cos_sim = np.dot(u_vec, t_vec)
                    # 转换为0-100的分数
                    vec_score = max(0, min(100, 50 * (cos_sim + 1)))
                    vec_scores.append(vec_score)
                
                if vec_scores:
                    shape_score = np.mean(vec_scores)
            
            # 3. 计算最终分数 (加权平均)
            if point_scores:
                point_weight = 0.7
                shape_weight = 0.3 if shape_score > 0 else 0
                
                # 调整权重
                total_weight = point_weight + shape_weight
                if total_weight > 0:
                    point_weight /= total_weight
                    shape_weight /= total_weight
                    
                    point_score = np.mean(point_scores)
                    score = point_weight * point_score + shape_weight * shape_score
                else:
                    score = 40
            else:
                score = 40
        
        # 添加评估详情
        result['details'] = {
            'computation_time': time.time() - start_time,
            'valid_points': len(valid_points),
            'total_points': len(keypoints),
            'method': 'simple' if low_quality else 'detailed'
        }
        
        # 确定状态和反馈
        score = max(0, min(100, score))
        result['score'] = int(score)
        
        if score >= self.feedback_thresholds['excellent']:
            result['status'] = 'CORRECT'
            result['feedback'] = '姿势非常标准，做得很好！'
            self.consecutive_correct += 1
        elif score >= self.feedback_thresholds['good']:
            result['status'] = 'CORRECT'
            result['feedback'] = '姿势良好，请保持！'
            self.consecutive_correct += 1
        elif score >= self.feedback_thresholds['moderate']:
            result['status'] = 'PARTIALLY_CORRECT'
            result['feedback'] = '姿势基本正确，可以稍作调整'
            self.consecutive_correct = 0
        elif score >= self.feedback_thresholds['poor']:
            result['status'] = 'INCORRECT'
            result['feedback'] = '请继续调整姿势以匹配轮廓'
            self.consecutive_correct = 0
        else:
            result['status'] = 'INCORRECT'
            result['feedback'] = '姿势不正确，请参考指导轮廓进行调整'
            self.consecutive_correct = 0
        
        # 检查是否达到连续正确要求
        if self.consecutive_correct >= self.required_correct:
            result['should_advance'] = True
            self.consecutive_correct = 0
        
        return result
    
    def overlay_feedback(self, frame, keypoints, result, pose_type):
        """在视频帧上叠加反馈信息
        
        Args:
            frame: 输入视频帧
            keypoints: 检测到的关键点
            result: 评估结果
            pose_type: 动作类型
            
        Returns:
            带有反馈信息的视频帧
        """
        if frame is None:
            return None
            
        # 创建展示用画布
        output = frame.copy()
        h, w = output.shape[:2]
        
        # 1. 添加动作模板轮廓
        if pose_type in self.pose_models:
            template = self.pose_models[pose_type].get('template')
            if template is not None:
                # 调整模板大小以匹配视频帧
                template_resized = cv2.resize(template, (w, h))
                
                # 基于评分设置轮廓透明度
                score = result.get('score', 0)
                alpha = max(0.1, min(0.5, 0.1 + (100 - score) / 100 * 0.4))
                
                # 叠加轮廓
                # 使用加权方法融合图像
                cv2.addWeighted(template_resized, alpha, output, 1 - alpha, 0, output)
        
        # 2. 可视化用户关键点
        if keypoints is not None and len(keypoints) > 0:
            # 使用不同颜色根据置信度绘制关键点
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.2:  # 只显示置信度足够的点
                    # 计算颜色 (绿色=高置信度，红色=低置信度)
                    color = (0, int(255 * conf), int(255 * (1 - conf)))
                    
                    # 绘制关键点
                    cv2.circle(output, (int(x), int(y)), 5, color, -1)
                    
                    # 连接相邻关键点形成骨架
                    if i > 0 and i < len(keypoints) and keypoints[i-1][2] > 0.2:
                        cv2.line(output, 
                                (int(keypoints[i-1][0]), int(keypoints[i-1][1])),
                                (int(x), int(y)),
                                color, 2)
        
        # 3. 添加状态文本反馈
        status = result.get('status', 'UNKNOWN')
        score = result.get('score', 0)
        feedback = result.get('feedback', '请调整姿势')
        
        # 状态指示器背景
        status_bg_color = (0, 0, 0)
        if status == 'CORRECT':
            status_bg_color = (0, 150, 0)  # 深绿色
        elif status == 'PARTIALLY_CORRECT':
            status_bg_color = (0, 120, 120)  # 蓝绿色
        elif status == 'INCORRECT':
            status_bg_color = (0, 0, 150)  # 深红色
        
        # 绘制状态背景
        cv2.rectangle(output, (0, 0), (w, 60), status_bg_color, -1)
        
        # 添加标题和得分
        cv2.putText(output, f"康复动作: {pose_type.replace('_', ' ').title()}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, f"分数: {score}", 
                   (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 添加反馈文本
        feedback_bg = (0, 0, 0, 150)  # 半透明黑色
        cv2.rectangle(output, (10, h - 60), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.putText(output, feedback, (20, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 4. 添加进度指示器
        if 'should_advance' in result and result['should_advance']:
            # 绘制"动作完成"提示
            cv2.rectangle(output, (int(w/2) - 150, int(h/2) - 50), 
                         (int(w/2) + 150, int(h/2) + 50), (0, 150, 0), -1)
            cv2.putText(output, "动作完成！即将进入下一步", 
                       (int(w/2) - 140, int(h/2) + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # 绘制进度条
            progress = min(1.0, max(0.0, self.consecutive_correct / self.required_correct))
            
            # 绘制进度条背景
            bar_h, bar_w = 15, w - 100
            bar_x, bar_y = 50, h - 80
            cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
            
            # 绘制当前进度
            filled_w = int(bar_w * progress)
            if filled_w > 0:
                # 颜色从红到绿渐变
                bar_color = (0, int(255 * progress), int(255 * (1 - progress)))
                cv2.rectangle(output, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_color, -1)
            
            # 进度文本
            cv2.putText(output, f"保持正确姿势: {self.consecutive_correct}/{self.required_correct}", 
                       (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                       
        return output 