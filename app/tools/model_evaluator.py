import os
import json
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys

# 添加父目录到路径，以便导入模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.spine_detector import SpineKeypointDetector


class ModelEvaluator:
    """脊柱关键点检测模型评估工具"""
    
    def __init__(self, model_path, test_image_dir, test_annotation_file, output_dir):
        """
        初始化评估工具
        
        Args:
            model_path: 模型文件路径
            test_image_dir: 测试图像目录
            test_annotation_file: 测试标注文件路径
            output_dir: 输出目录
        """
        self.model_path = Path(model_path)
        self.test_image_dir = Path(test_image_dir)
        self.test_annotation_file = Path(test_annotation_file)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载测试标注
        with open(self.test_annotation_file, 'r') as f:
            self.test_annotations = json.load(f)
            
        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
        # 评估结果
        self.results = {
            'per_image': {},
            'overall': {}
        }
        
    def _load_model(self):
        """加载模型"""
        print(f"加载模型: {self.model_path}")
        
        try:
            # 尝试加载TorchScript模型
            model = torch.jit.load(str(self.model_path), map_location=self.device)
            print("已加载TorchScript模型")
            return model
        except Exception as e:
            print(f"无法加载TorchScript模型: {e}")
            
            try:
                # 尝试加载状态字典
                model = SpineKeypointDetector()
                state_dict = torch.load(str(self.model_path), map_location=self.device)
                
                # 如果状态字典包含在'state_dict'键中
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                    
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                print("已加载状态字典模型")
                return model
            except Exception as e:
                print(f"无法加载状态字典模型: {e}")
                raise ValueError(f"无法加载模型: {self.model_path}")
    
    def evaluate(self):
        """评估模型性能"""
        print("开始评估模型性能...")
        
        # 用于计算整体指标的数据
        all_pred_keypoints = []
        all_true_keypoints = []
        all_distances = []
        all_pck_values = []  # PCK: Percentage of Correct Keypoints
        
        # 评估每张图像
        for img_file, annotation in tqdm(self.test_annotations.items(), desc="评估图像"):
            # 读取图像
            img_path = self.test_image_dir / img_file
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue
                
            # 获取真实关键点
            true_keypoints = np.array(annotation['keypoints'])
            
            # 预测关键点
            pred_keypoints = self._predict_keypoints(image)
            
            # 计算评估指标
            metrics = self._calculate_metrics(pred_keypoints, true_keypoints, image.shape[:2])
            
            # 保存结果
            self.results['per_image'][img_file] = metrics
            
            # 累积数据用于计算整体指标
            all_pred_keypoints.append(pred_keypoints)
            all_true_keypoints.append(true_keypoints)
            all_distances.extend(metrics['distances'])
            all_pck_values.append(metrics['pck'])
            
            # 可视化结果
            self._visualize_results(image, true_keypoints, pred_keypoints, img_file)
        
        # 计算整体指标
        self.results['overall'] = {
            'mean_distance': np.mean(all_distances),
            'median_distance': np.median(all_distances),
            'std_distance': np.std(all_distances),
            'mean_pck': np.mean(all_pck_values),
            'rmse': np.sqrt(mean_squared_error(
                np.vstack([kp[:, :2].flatten() for kp in all_true_keypoints]),
                np.vstack([kp[:, :2].flatten() for kp in all_pred_keypoints])
            ))
        }
        
        # 保存评估结果
        self._save_results()
        
        # 生成评估报告
        self._generate_report()
        
        return self.results
    
    def _predict_keypoints(self, image):
        """预测图像中的关键点"""
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 预处理图像
        h, w = image_rgb.shape[:2]
        input_size = 512  # 假设模型输入大小为512x512
        
        # 调整图像大小
        image_resized = cv2.resize(image_rgb, (input_size, input_size))
        
        # 归一化
        image_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.device))
            
        # 处理输出
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # 有些模型可能返回多个输出
            
        # 转换为numpy数组
        keypoints = outputs.cpu().numpy()[0]  # [num_keypoints, 3]
        
        # 缩放回原始图像大小
        keypoints[:, 0] *= w / input_size
        keypoints[:, 1] *= h / input_size
        
        return keypoints
    
    def _calculate_metrics(self, pred_keypoints, true_keypoints, image_shape):
        """计算评估指标"""
        h, w = image_shape
        
        # 计算每个关键点的欧氏距离
        distances = []
        for i in range(min(len(pred_keypoints), len(true_keypoints))):
            if true_keypoints[i, 2] > 0:  # 只评估可见的关键点
                dist = np.sqrt(
                    (pred_keypoints[i, 0] - true_keypoints[i, 0]) ** 2 +
                    (pred_keypoints[i, 1] - true_keypoints[i, 1]) ** 2
                )
                distances.append(dist)
        
        # 计算PCK (Percentage of Correct Keypoints)
        # 阈值通常设置为图像对角线长度的一定比例
        threshold = 0.05 * np.sqrt(h**2 + w**2)
        pck = np.mean([1 if d <= threshold else 0 for d in distances]) if distances else 0
        
        # 计算RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean(np.array(distances) ** 2)) if distances else 0
        
        return {
            'distances': distances,
            'mean_distance': np.mean(distances) if distances else 0,
            'median_distance': np.median(distances) if distances else 0,
            'max_distance': np.max(distances) if distances else 0,
            'pck': pck,
            'rmse': rmse
        }
    
    def _visualize_results(self, image, true_keypoints, pred_keypoints, img_file):
        """可视化评估结果"""
        # 创建可视化图像
        vis_image = image.copy()
        
        # 绘制真实关键点（绿色）
        for i, kp in enumerate(true_keypoints):
            if kp[2] > 0:  # 只绘制可见的关键点
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(vis_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(vis_image, str(i+1), (x+5, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制预测关键点（红色）
        for i, kp in enumerate(pred_keypoints):
            if i < len(true_keypoints) and true_keypoints[i, 2] > 0:  # 只绘制对应于可见真实关键点的预测
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(vis_image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(vis_image, str(i+1), (x+5, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # 绘制连接线
                true_x, true_y = int(true_keypoints[i, 0]), int(true_keypoints[i, 1])
                cv2.line(vis_image, (x, y), (true_x, true_y), (255, 0, 255), 1)
        
        # 保存可视化结果
        output_path = self.output_dir / f"vis_{img_file}"
        cv2.imwrite(str(output_path), vis_image)
    
    def _save_results(self):
        """保存评估结果"""
        # 保存为JSON文件
        output_path = self.output_dir / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)
            
        print(f"评估结果已保存到: {output_path}")
    
    def _generate_report(self):
        """生成评估报告"""
        # 提取整体指标
        overall = self.results['overall']
        
        # 创建报告文件
        report_path = self.output_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write("脊柱关键点检测模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("整体性能指标:\n")
            f.write(f"平均距离误差: {overall['mean_distance']:.2f} 像素\n")
            f.write(f"中位数距离误差: {overall['median_distance']:.2f} 像素\n")
            f.write(f"距离标准差: {overall['std_distance']:.2f} 像素\n")
            f.write(f"PCK (正确关键点百分比): {overall['mean_pck']*100:.2f}%\n")
            f.write(f"RMSE (均方根误差): {overall['rmse']:.2f} 像素\n\n")
            
            f.write("每张图像的性能:\n")
            for img_file, metrics in self.results['per_image'].items():
                f.write(f"图像: {img_file}\n")
                f.write(f"  平均距离误差: {metrics['mean_distance']:.2f} 像素\n")
                f.write(f"  PCK: {metrics['pck']*100:.2f}%\n")
                f.write(f"  RMSE: {metrics['rmse']:.2f} 像素\n")
                f.write("\n")
        
        print(f"评估报告已生成: {report_path}")
        
        # 生成距离误差直方图
        self._plot_distance_histogram()
        
        # 生成PCK曲线
        self._plot_pck_curve()
    
    def _plot_distance_histogram(self):
        """绘制距离误差直方图"""
        # 收集所有距离
        all_distances = []
        for metrics in self.results['per_image'].values():
            all_distances.extend(metrics['distances'])
        
        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.hist(all_distances, bins=30, alpha=0.7, color='blue')
        plt.axvline(np.mean(all_distances), color='red', linestyle='dashed', linewidth=1, 
                   label=f'平均值: {np.mean(all_distances):.2f}')
        plt.axvline(np.median(all_distances), color='green', linestyle='dashed', linewidth=1,
                   label=f'中位数: {np.median(all_distances):.2f}')
        
        plt.xlabel('距离误差 (像素)')
        plt.ylabel('频率')
        plt.title('关键点距离误差分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        plt.savefig(str(self.output_dir / "distance_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pck_curve(self):
        """绘制PCK曲线"""
        # 收集所有距离
        all_distances = []
        for metrics in self.results['per_image'].values():
            all_distances.extend(metrics['distances'])
        
        # 计算不同阈值下的PCK
        thresholds = np.linspace(0, 50, 100)  # 0到50像素的阈值
        pck_values = []
        
        for threshold in thresholds:
            pck = np.mean([1 if d <= threshold else 0 for d in all_distances])
            pck_values.append(pck)
        
        # 绘制PCK曲线
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, pck_values, 'b-', linewidth=2)
        
        # 标记一些关键点
        for t in [5, 10, 20]:
            idx = np.argmin(np.abs(thresholds - t))
            plt.plot(thresholds[idx], pck_values[idx], 'ro')
            plt.text(thresholds[idx]+1, pck_values[idx], 
                    f'阈值={t}px: {pck_values[idx]*100:.1f}%', 
                    verticalalignment='bottom')
        
        plt.xlabel('距离阈值 (像素)')
        plt.ylabel('PCK (正确关键点百分比)')
        plt.title('PCK曲线')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        # 保存图表
        plt.savefig(str(self.output_dir / "pck_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()


class NumpyEncoder(json.JSONEncoder):
    """处理NumPy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='脊柱关键点检测模型评估工具')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--test_dir', type=str, required=True, help='测试图像目录')
    parser.add_argument('--test_annotation', type=str, required=True, help='测试标注文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    # 创建并运行评估工具
    evaluator = ModelEvaluator(
        args.model,
        args.test_dir,
        args.test_annotation,
        args.output_dir
    )
    results = evaluator.evaluate()
    
    # 打印整体结果
    print("\n整体评估结果:")
    print(f"平均距离误差: {results['overall']['mean_distance']:.2f} 像素")
    print(f"PCK (正确关键点百分比): {results['overall']['mean_pck']*100:.2f}%")
    print(f"RMSE (均方根误差): {results['overall']['rmse']:.2f} 像素")


if __name__ == '__main__':
    main() 