import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path


class KeypointAnnotator:
    """脊柱关键点标注工具"""
    
    def __init__(self, image_dir, output_file, num_keypoints=17):
        """
        初始化标注工具
        
        Args:
            image_dir: 图像目录
            output_file: 输出标注文件路径
            num_keypoints: 关键点数量
        """
        self.image_dir = Path(image_dir)
        self.output_file = Path(output_file)
        self.num_keypoints = num_keypoints
        
        # 获取图像文件列表
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not self.image_files:
            print(f"错误: 在 {image_dir} 中没有找到图像文件")
            sys.exit(1)
            
        # 加载现有标注（如果存在）
        self.annotations = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    self.annotations = json.load(f)
                print(f"已加载 {len(self.annotations)} 个现有标注")
            except Exception as e:
                print(f"警告: 无法加载现有标注文件: {e}")
        
        # 当前图像索引和标注状态
        self.current_idx = 0
        self.current_keypoint = 0
        self.keypoints = []
        self.window_name = "脊柱关键点标注工具"
        
        # 界面设置
        self.zoom_factor = 1.0
        self.offset_x, self.offset_y = 0, 0
        self.dragging = False
        self.last_x, self.last_y = 0, 0
        
        # 颜色设置
        self.colors = [
            (0, 0, 255),    # 红色
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 255, 255),  # 黄色
            (255, 0, 255),  # 紫色
            (255, 255, 0),  # 青色
        ]
        
    def load_image(self, idx):
        """加载指定索引的图像"""
        if 0 <= idx < len(self.image_files):
            self.current_idx = idx
            img_file = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_file)
            
            # 读取图像
            self.image = cv2.imread(img_path)
            if self.image is None:
                print(f"错误: 无法读取图像 {img_path}")
                return False
                
            # 调整图像大小以适应屏幕
            h, w = self.image.shape[:2]
            screen_h, screen_w = 900, 1600  # 假设的屏幕尺寸
            
            if h > screen_h or w > screen_w:
                scale = min(screen_h / h, screen_w / w)
                self.image = cv2.resize(self.image, None, fx=scale, fy=scale)
            
            # 重置缩放和平移
            self.zoom_factor = 1.0
            self.offset_x, self.offset_y = 0, 0
            
            # 加载现有标注或创建新标注
            if img_file in self.annotations:
                self.keypoints = self.annotations[img_file]['keypoints']
                self.current_keypoint = min(len(self.keypoints), self.num_keypoints)
                print(f"已加载图像 {img_file} 的现有标注")
            else:
                self.keypoints = []
                self.current_keypoint = 0
                
            return True
        return False
        
    def save_annotations(self):
        """保存标注到文件"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # 保存标注
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
            
        print(f"已保存 {len(self.annotations)} 个标注到 {self.output_file}")
        
    def draw_keypoints(self, img):
        """在图像上绘制关键点"""
        h, w = img.shape[:2]
        
        # 绘制已标注的关键点
        for i, kp in enumerate(self.keypoints):
            x, y = int(kp[0]), int(kp[1])
            color = self.colors[i % len(self.colors)]
            
            # 绘制关键点
            cv2.circle(img, (x, y), 5, color, -1)
            
            # 绘制关键点编号
            cv2.putText(img, str(i+1), (x+5, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 如果有多个关键点，绘制连接线
            if i > 0:
                prev_x, prev_y = int(self.keypoints[i-1][0]), int(self.keypoints[i-1][1])
                cv2.line(img, (prev_x, prev_y), (x, y), color, 2)
        
        # 显示当前状态
        status = f"图像: {self.current_idx+1}/{len(self.image_files)} | "
        status += f"关键点: {self.current_keypoint}/{self.num_keypoints} | "
        status += f"缩放: {self.zoom_factor:.1f}x"
        
        cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示帮助信息
        help_text = [
            "按键说明:",
            "左键点击: 标注关键点",
            "右键点击: 删除最后一个关键点",
            "鼠标滚轮: 缩放图像",
            "鼠标拖动: 平移图像",
            "S: 保存当前标注",
            "N: 下一张图像",
            "P: 上一张图像",
            "R: 重置当前标注",
            "Q/ESC: 退出程序"
        ]
        
        for i, text in enumerate(help_text):
            cv2.putText(img, text, (10, h - 20 - i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调函数"""
        # 计算实际坐标（考虑缩放和平移）
        real_x = int((x - self.offset_x) / self.zoom_factor)
        real_y = int((y - self.offset_y) / self.zoom_factor)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键点击：添加关键点
            if self.current_keypoint < self.num_keypoints:
                # 添加新关键点
                if self.current_keypoint < len(self.keypoints):
                    self.keypoints[self.current_keypoint] = [real_x, real_y, 1.0]  # x, y, visibility
                else:
                    self.keypoints.append([real_x, real_y, 1.0])
                
                self.current_keypoint += 1
                
                # 如果完成所有关键点标注，自动保存
                if self.current_keypoint == self.num_keypoints:
                    self.save_current_annotation()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键点击：删除最后一个关键点
            if self.current_keypoint > 0:
                self.current_keypoint -= 1
                if self.current_keypoint < len(self.keypoints):
                    self.keypoints.pop()
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            # 鼠标滚轮：缩放
            if flags > 0:  # 向上滚动，放大
                self.zoom_factor *= 1.1
            else:  # 向下滚动，缩小
                self.zoom_factor /= 1.1
                
            # 限制缩放范围
            self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))
            
        elif event == cv2.EVENT_MBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY):
            # 中键按下或Ctrl+左键：开始拖动
            self.dragging = True
            self.last_x, self.last_y = x, y
            
        elif event == cv2.EVENT_MBUTTONUP or (event == cv2.EVENT_LBUTTONUP and flags & cv2.EVENT_FLAG_CTRLKEY):
            # 中键释放或Ctrl+左键释放：停止拖动
            self.dragging = False
            
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # 鼠标移动且正在拖动：平移图像
            dx, dy = x - self.last_x, y - self.last_y
            self.offset_x += dx
            self.offset_y += dy
            self.last_x, self.last_y = x, y
    
    def save_current_annotation(self):
        """保存当前图像的标注"""
        img_file = self.image_files[self.current_idx]
        
        # 获取原始图像尺寸
        img_path = os.path.join(self.image_dir, img_file)
        orig_img = cv2.imread(img_path)
        h, w = orig_img.shape[:2]
        
        # 调整关键点坐标到原始图像尺寸
        adjusted_keypoints = []
        for kp in self.keypoints:
            # 复制关键点以避免修改原始数据
            adjusted_kp = kp.copy()
            
            # 如果图像被调整过大小，需要重新缩放关键点坐标
            if self.image.shape[:2] != (h, w):
                scale_x = w / self.image.shape[1]
                scale_y = h / self.image.shape[0]
                adjusted_kp[0] *= scale_x
                adjusted_kp[1] *= scale_y
                
            adjusted_keypoints.append(adjusted_kp)
        
        # 保存标注
        self.annotations[img_file] = {
            'keypoints': adjusted_keypoints,
            'image_width': w,
            'image_height': h
        }
        
        print(f"已保存图像 {img_file} 的标注")
        
    def run(self):
        """运行标注工具"""
        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # 加载第一张图像
        if not self.load_image(0):
            print("错误: 无法加载图像")
            return
            
        while True:
            # 创建显示图像的副本
            display = self.image.copy()
            
            # 应用缩放和平移
            if self.zoom_factor != 1.0 or self.offset_x != 0 or self.offset_y != 0:
                h, w = display.shape[:2]
                M = np.float32([[self.zoom_factor, 0, self.offset_x], 
                                [0, self.zoom_factor, self.offset_y]])
                display = cv2.warpAffine(display, M, (w, h))
            
            # 绘制关键点
            display = self.draw_keypoints(display)
            
            # 显示图像
            cv2.imshow(self.window_name, display)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC 或 q: 退出
                break
                
            elif key == ord('s'):  # s: 保存当前标注
                self.save_current_annotation()
                self.save_annotations()
                
            elif key == ord('n'):  # n: 下一张图像
                if self.current_keypoint == self.num_keypoints or self.current_keypoint == 0:
                    # 只有完成所有关键点标注或未开始标注时才能切换图像
                    if self.current_keypoint == self.num_keypoints:
                        self.save_current_annotation()
                    self.load_image(self.current_idx + 1)
                else:
                    print("警告: 请先完成当前图像的标注或重置标注")
                    
            elif key == ord('p'):  # p: 上一张图像
                if self.current_keypoint == self.num_keypoints or self.current_keypoint == 0:
                    # 只有完成所有关键点标注或未开始标注时才能切换图像
                    if self.current_keypoint == self.num_keypoints:
                        self.save_current_annotation()
                    self.load_image(self.current_idx - 1)
                else:
                    print("警告: 请先完成当前图像的标注或重置标注")
                    
            elif key == ord('r'):  # r: 重置当前标注
                self.keypoints = []
                self.current_keypoint = 0
                
        # 保存标注并关闭窗口
        self.save_annotations()
        cv2.destroyAllWindows()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='脊柱关键点标注工具')
    parser.add_argument('--image_dir', type=str, required=True, help='图像目录路径')
    parser.add_argument('--output', type=str, required=True, help='输出标注文件路径')
    parser.add_argument('--num_keypoints', type=int, default=17, help='关键点数量')
    
    args = parser.parse_args()
    
    # 创建并运行标注工具
    annotator = KeypointAnnotator(args.image_dir, args.output, args.num_keypoints)
    annotator.run()


if __name__ == '__main__':
    main() 