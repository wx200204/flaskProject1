import numpy as np
import cv2
from scipy.spatial import distance

class SpinePostProcessor:
    """基于解剖学约束的脊柱后处理"""
    
    def __init__(self, max_curve_angle=15, min_vertebra_dist=10):
        """
        Args:
            max_curve_angle: 最大允许的脊柱弯曲角度(度)
            min_vertebra_dist: 相邻椎骨最小距离(像素)
        """
        self.max_curve_angle = max_curve_angle
        self.min_vertebra_dist = min_vertebra_dist

    def apply_constraints(self, keypoints, image_shape):
        """应用解剖学约束"""
        # 1. 过滤低置信度关键点
        valid_points = keypoints[keypoints[:,2] > 0.5][:,:2] if keypoints.shape[1] > 2 else keypoints
        
        if len(valid_points) < 4:  # 需要至少4个点进行有意义的处理
            return valid_points, 0.0, 0.0
        
        # 2. 按y坐标排序（从上到下）
        valid_points = valid_points[np.argsort(valid_points[:, 1])]
        
        # 3. 强制最小椎骨间距
        valid_points = self._enforce_min_distance(valid_points)
        
        # 4. 检查椎体排列方向一致性
        valid_points = self._check_orientation(valid_points)
        
        # 5. 平滑脊柱曲线
        smoothed_points = self._smooth_curve(valid_points)
        
        # 6. 计算解剖学合理性评分
        symmetry_score = self._calc_symmetry(smoothed_points, image_shape)
        curvature_score = self._calc_curvature(smoothed_points)
        
        return smoothed_points, symmetry_score, curvature_score

    def _enforce_min_distance(self, points):
        """动态间距约束（基于图像尺寸）"""
        if len(points) < 2:
            return points
            
        # 动态计算最小间距（图像高度的2%）
        min_dist = max(self.min_vertebra_dist, int(0.02 * (np.max(points[:,1]) - np.min(points[:,1]))))
        
        filtered = [points[0]]
        for pt in points[1:]:
            # 添加水平方向约束：相邻点水平偏移不超过垂直间距的2倍
            vertical_dist = abs(pt[1] - filtered[-1][1])
            if vertical_dist >= min_dist and \
               abs(pt[0] - filtered[-1][0]) < 2 * vertical_dist:
                filtered.append(pt)
        return np.array(filtered)

    def _check_orientation(self, points):
        """校验椎体排列方向一致性"""
        if len(points) < 3:
            return points
        
        import math
        
        # 计算相邻点角度变化
        angles = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            angles.append(math.degrees(math.atan2(dy, dx)))
        
        # 过滤突变角度(超过30度)
        filtered = [points[0]]
        angle_mean = angles[0]  # 初始平均值
        
        for i in range(1, len(points)-1):
            # 使用移动平均
            angle_diff = abs(angles[i] - angle_mean)
            if angle_diff < 30:
                filtered.append(points[i+1])
                # 更新移动平均
                angle_mean = 0.8 * angle_mean + 0.2 * angles[i]
        
        # 确保至少保留一些点
        if len(filtered) < 3 and len(points) >= 3:
            return points
            
        return np.array(filtered)

    def _smooth_curve(self, points, window_size=3):
        """滑动平均平滑脊柱曲线"""
        if len(points) < window_size:
            return points
            
        # 使用1D卷积进行平滑处理
        kernel = np.ones(window_size)/window_size
        
        # 分别平滑x和y坐标
        x_smooth = np.convolve(points[:,0], kernel, mode='same')
        y_smooth = np.convolve(points[:,1], kernel, mode='same')
        
        # 保持端点不变以避免边缘效应
        half_win = window_size // 2
        if len(points) > window_size:
            x_smooth[:half_win] = points[:half_win, 0]
            x_smooth[-half_win:] = points[-half_win:, 0]
            y_smooth[:half_win] = points[:half_win, 1]
            y_smooth[-half_win:] = points[-half_win:, 1]
        
        return np.column_stack([x_smooth, y_smooth])

    def _calc_symmetry(self, points, image_shape):
        """计算脊柱对称性评分"""
        if len(points) < 3:
            return 0.0
        
        mid_x = image_shape[1] // 2
        
        # 计算点与中线的偏差
        deviations = [abs(pt[0] - mid_x) / mid_x for pt in points]
        
        # 考虑偏差的位置-相邻点偏向同一侧会导致更低的对称性
        same_side_penalty = 0
        for i in range(1, len(points)):
            if (points[i][0] > mid_x and points[i-1][0] > mid_x) or \
               (points[i][0] < mid_x and points[i-1][0] < mid_x):
                same_side_penalty += 0.1
        
        # 对称性分数介于0-1之间，1为完全对称
        return max(0.0, min(1.0, 1 - (np.mean(deviations) + same_side_penalty)))

    def _calc_curvature(self, points):
        """计算脊柱曲率合理性评分"""
        if len(points) < 3:
            return 1.0
        
        # 计算相邻点之间的矢量
        vectors = np.diff(points, axis=0)
        
        # 归一化矢量
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 避免除零错误
        normalized_vectors = vectors / norms
        
        # 计算相邻向量之间的角度
        angles = []
        for i in range(1, len(normalized_vectors)):
            cos_theta = np.clip(
                np.dot(normalized_vectors[i-1], normalized_vectors[i]), -1.0, 1.0
            )
            angles.append(np.degrees(np.arccos(cos_theta)))
        
        if not angles:
            return 1.0
            
        # 计算角度方差和平均值
        avg_angle = np.mean(angles)
        angle_var = np.var(angles)
        
        # 平滑度分数：基于角度方差和平均角度
        smoothness = 1.0 - min(1.0, avg_angle / self.max_curve_angle)
        variance_penalty = min(0.5, angle_var / 100)
        
        return max(0.0, smoothness - variance_penalty)

    def visualize_postprocess(self, image, raw_points, processed_points):
        """可视化后处理效果"""
        vis_img = image.copy()
        
        # 绘制原始点
        for pt in raw_points:
            point = pt[:2] if len(pt) > 2 else pt
            cv2.circle(vis_img, tuple(map(int, point)), 3, (0,0,255), -1)
        
        # 绘制处理后点
        for pt in processed_points:
            cv2.circle(vis_img, tuple(map(int, pt)), 5, (255,0,0), -1)
            
        # 绘制处理后曲线
        for i in range(1, len(processed_points)):
            cv2.line(vis_img, 
                    tuple(map(int, processed_points[i-1])),
                    tuple(map(int, processed_points[i])),
                    (0,255,0), 2)
                    
        # 添加说明文字
        cv2.putText(vis_img, "原始点 (红色)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(vis_img, "处理后点 (蓝色)", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(vis_img, "脊柱曲线 (绿色)", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                   
        return vis_img