# 脊柱关键点检测工具集

本目录包含用于脊柱关键点检测的各种工具，包括数据标注、数据增强、模型训练和评估等。

## 目录

1. [环境要求](#环境要求)
2. [数据标注工具](#数据标注工具)
3. [数据增强工具](#数据增强工具)
4. [模型训练](#模型训练)
5. [模型评估](#模型评估)
6. [工作流程示例](#工作流程示例)

## 环境要求

使用这些工具需要安装以下依赖：

```bash
pip install torch torchvision numpy opencv-python matplotlib scikit-learn tqdm
```

## 数据标注工具

`keypoint_annotator.py` 是一个用于标注脊柱X光片关键点的工具。

### 使用方法

```bash
python keypoint_annotator.py --image_dir <图像目录> --output <输出标注文件> --num_keypoints <关键点数量>
```

参数说明：
- `--image_dir`: 包含脊柱X光片的目录
- `--output`: 输出标注文件的路径（JSON格式）
- `--num_keypoints`: 每个脊柱需要标注的关键点数量，默认为17

### 操作说明

- 左键点击：标注关键点
- 右键点击：删除最后一个关键点
- 鼠标滚轮：缩放图像
- 鼠标拖动：平移图像
- S键：保存当前标注
- N键：下一张图像
- P键：上一张图像
- R键：重置当前标注
- Q/ESC键：退出程序

## 数据增强工具

`data_augmentation.py` 用于对标注好的脊柱X光片数据集进行增强，生成更多的训练样本。

### 使用方法

```bash
python data_augmentation.py --image_dir <原始图像目录> --annotation_file <原始标注文件> --output_dir <输出图像目录> --output_annotation <输出标注文件> --num_augmentations <增强数量>
```

参数说明：
- `--image_dir`: 原始图像目录
- `--annotation_file`: 原始标注文件路径
- `--output_dir`: 增强后图像的输出目录
- `--output_annotation`: 增强后标注的输出文件路径
- `--num_augmentations`: 每张图像生成的增强版本数量，默认为5

### 增强方法

该工具包含以下增强方法：
- 随机旋转
- 随机调整亮度和对比度
- 随机添加噪声
- 水平翻转
- 随机缩放

## 模型训练

`../models/train_model.py` 用于训练脊柱关键点检测模型。

### 使用方法

```bash
python ../models/train_model.py --data_dir <图像目录> --train_annotations <训练标注文件> --val_annotations <验证标注文件> --output_dir <输出目录> --batch_size <批量大小> --epochs <训练轮数> --learning_rate <学习率> --num_keypoints <关键点数量> --pretrained <预训练模型>
```

参数说明：
- `--data_dir`: 包含训练和验证图像的目录
- `--train_annotations`: 训练集标注文件路径
- `--val_annotations`: 验证集标注文件路径
- `--output_dir`: 模型输出目录，默认为 `./output`
- `--batch_size`: 训练批量大小，默认为8
- `--epochs`: 训练轮数，默认为50
- `--learning_rate`: 学习率，默认为0.001
- `--num_keypoints`: 关键点数量，默认为17
- `--pretrained`: 预训练模型路径（可选）

### 输出文件

训练过程会生成以下文件：
- `best_model.pth`: 验证集上性能最好的模型
- `model_epoch_N.pth`: 每个训练轮次的模型
- `model_scripted.pth`: 导出的TorchScript模型，可用于生产环境

## 模型评估

`model_evaluator.py` 用于评估训练好的脊柱关键点检测模型的性能。

### 使用方法

```bash
python model_evaluator.py --model <模型文件> --test_dir <测试图像目录> --test_annotation <测试标注文件> --output_dir <输出目录>
```

参数说明：
- `--model`: 模型文件路径
- `--test_dir`: 测试图像目录
- `--test_annotation`: 测试标注文件路径
- `--output_dir`: 评估结果输出目录

### 评估指标

评估工具会计算以下指标：
- 平均距离误差
- 中位数距离误差
- PCK (Percentage of Correct Keypoints)
- RMSE (Root Mean Square Error)

### 输出文件

评估过程会生成以下文件：
- `evaluation_results.json`: 详细的评估结果
- `evaluation_report.txt`: 评估报告
- `distance_histogram.png`: 距离误差直方图
- `pck_curve.png`: PCK曲线
- `vis_*.jpg`: 可视化结果，显示真实关键点和预测关键点

## 工作流程示例

以下是使用这些工具的完整工作流程示例：

### 1. 数据标注

```bash
# 创建标注目录
mkdir -p data/annotations

# 标注脊柱X光片
python keypoint_annotator.py --image_dir data/raw_images --output data/annotations/spine_keypoints.json --num_keypoints 17
```

### 2. 数据增强

```bash
# 创建增强数据目录
mkdir -p data/augmented

# 增强数据集
python data_augmentation.py --image_dir data/raw_images --annotation_file data/annotations/spine_keypoints.json --output_dir data/augmented --output_annotation data/annotations/augmented_keypoints.json --num_augmentations 5
```

### 3. 数据集划分

```bash
# 使用脚本划分数据集（需要自行创建）
python split_dataset.py --annotation_file data/annotations/augmented_keypoints.json --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --output_dir data/annotations
```

### 4. 模型训练

```bash
# 创建模型输出目录
mkdir -p models/output

# 训练模型
python ../models/train_model.py --data_dir data/augmented --train_annotations data/annotations/train.json --val_annotations data/annotations/val.json --output_dir models/output --batch_size 16 --epochs 100 --learning_rate 0.001 --num_keypoints 17
```

### 5. 模型评估

```bash
# 创建评估输出目录
mkdir -p evaluation

# 评估模型
python model_evaluator.py --model models/output/best_model.pth --test_dir data/augmented --test_annotation data/annotations/test.json --output_dir evaluation
```

### 6. 在应用中使用模型

将训练好的模型（`models/output/model_scripted.pth`）复制到应用的模型目录中，即可在应用中使用。

```bash
cp models/output/model_scripted.pth ../../models/spine_keypoint_detector.pth
``` 