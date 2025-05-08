import os
import json
import argparse
import random
from pathlib import Path


def split_dataset(annotation_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, output_dir=None, random_seed=42):
    """
    将标注数据集划分为训练集、验证集和测试集
    
    Args:
        annotation_file: 标注文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        output_dir: 输出目录
        random_seed: 随机种子
    
    Returns:
        train_data, val_data, test_data: 划分后的数据集
    """
    # 检查比例和是否为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例之和必须为1"
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 加载标注文件
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # 获取所有图像文件名
    image_files = list(annotations.keys())
    
    # 随机打乱
    random.shuffle(image_files)
    
    # 计算划分点
    train_end = int(len(image_files) * train_ratio)
    val_end = train_end + int(len(image_files) * val_ratio)
    
    # 划分数据集
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    # 创建划分后的数据集
    train_data = {img: annotations[img] for img in train_files}
    val_data = {img: annotations[img] for img in val_files}
    test_data = {img: annotations[img] for img in test_files}
    
    # 如果指定了输出目录，保存划分后的数据集
    if output_dir:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_dir / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
            
        with open(output_dir / "val.json", 'w') as f:
            json.dump(val_data, f, indent=2)
            
        with open(output_dir / "test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"数据集划分完成，已保存到 {output_dir}")
        print(f"训练集: {len(train_data)} 张图像")
        print(f"验证集: {len(val_data)} 张图像")
        print(f"测试集: {len(test_data)} 张图像")
    
    return train_data, val_data, test_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据集划分工具')
    parser.add_argument('--annotation_file', type=str, required=True, help='标注文件路径')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 划分数据集
    split_dataset(
        args.annotation_file,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.output_dir,
        args.random_seed
    )


if __name__ == '__main__':
    main() 