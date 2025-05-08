import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def draw_keypoints(image, keypoints, color=(0, 255, 0), radius=3):
    """在图像上绘制关键点
    
    Args:
        image: 输入图像
        keypoints: 关键点坐标列表
        color: 关键点颜色
        radius: 关键点半径
    
    Returns:
        标注后的图像
    """
    img = image.copy()
    for point in keypoints:
        x, y = int(point[0]), int(point[1])
        cv2.circle(img, (x, y), radius, color, -1)
    return img

from scipy.interpolate import splprep, splev

def draw_lines(image, keypoints, color=(0, 0, 255), thickness=2):
    """在图像上绘制连接线
    
    Args:
        image: 输入图像
        keypoints: 关键点坐标列表
        color: 线条颜色
        thickness: 线条粗细
    
    Returns:
        标注后的图像
    """
    img = image.copy()
    # B样条插值平滑
    if len(keypoints) > 3:
        try:
            tck, u = splprep(keypoints.T, s=0, per=False)
            u_new = np.linspace(u.min(), u.max(), 100)
            x_new, y_new = splev(u_new, tck, der=0)
            points = np.column_stack((x_new, y_new)).astype(np.int32)
        except:
            points = keypoints.astype(np.int32)
    else:
        points = keypoints.astype(np.int32)

    # 绘制平滑曲线
    cv2.polylines(img, [points], isClosed=False, color=color, thickness=thickness)
    return img

def add_angle_text(image, angle, position, color=(255, 255, 255)):
    """在图像上添加角度文本
    
    Args:
        image: 输入图像
        angle: 角度值
        position: 文本位置 (x, y)
        color: 文本颜色
    
    Returns:
        标注后的图像
    """
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Cobb Angle: {angle:.1f}°"
    cv2.putText(img, text, position, font, 1, color, 2, cv2.LINE_AA)
    return img

class MedicalVisualizer:
    """医学影像可视化引擎"""

    def __init__(self, colormap='viridis'):
        self.cmap = plt.get_cmap(colormap)

    def plot_spine_curvature(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """绘制脊柱曲率热力图"""
        # 曲线拟合
        x = keypoints[:, 0].astype(float)
        y = keypoints[:, 1].astype(float)
        coeffs = np.polyfit(y, x, 3)
        poly = np.poly1d(coeffs)
        y_new = np.linspace(y.min(), y.max(), 100)
        x_new = poly(y_new)

        # 生成热力网格
        h, w = image.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        dist = np.abs(grid_x - poly(grid_y))
        heatmap = np.exp(-dist ** 2 / (2 * (w / 20) ** 2))

        # 融合显示
        overlay = self.cmap(heatmap)[..., :3] * 255
        blended = cv2.addWeighted(image, 0.7, overlay.astype(np.uint8), 0.3, 0)
        return blended

    def generate_report_image(self, image: np.ndarray, keypoints: np.ndarray, angle: float) -> bytes:
        """生成PDF报告用示意图"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # 绘制原始影像
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 绘制关键点连线
        ax.plot(keypoints[:, 0], keypoints[:, 1], 'yo-', linewidth=2, markersize=8)

        # 标注Cobb角
        max_point = keypoints[np.argmax(keypoints[:, 1])]
        ax.text(max_point[0] + 50, max_point[1] - 30,
                f"Cobb Angle: {angle:.1f}°",
                color='white', fontsize=12,
                bbox=dict(facecolor='red', alpha=0.7))

        # 渲染为图像字节
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.close(fig)
        return cv2.imencode('.png', img)[1].tobytes()