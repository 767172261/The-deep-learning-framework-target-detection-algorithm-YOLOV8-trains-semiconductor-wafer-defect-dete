深度学习框架目标检测算法YOLOV8训练半导体晶圆缺陷检测数据集 通过训练的权重推理识别中心、甜甜圈、边缘位置、边缘环、位置、接近满、无、随机、划痕。
@[toc]

**半导体晶圆缺陷检测数据集，9类缺陷标签
数据集描述:
半导体晶圆缺陷检测数据集，9类缺陷标签：中心、甜甜圈、边缘位置、边缘环、位置、接近满、无、随机、划痕。包含训练集10400张，验证集1300张，测试集1300张图片，YOLO标注格式**

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7a8e458d17674685b0ba1bb50d5104ec.png)

**半导体晶圆缺陷检测数据集**（9类缺陷标签）使用 **YOLOv8** 进行训练、推理、评估的完整代码示例，仅供参考指南，包括从环境搭建到模型部署的全流程。

---

## ✅ 一、硬件与软件环境准备



### 2. 安装 CUDA 驱动
确保已安装支持 CUDA 的 NVIDIA 显卡驱动：

```bash
nvidia-smi
```

 [NVIDIA驱动官网](https://www.nvidia.cn/Download/index.aspx) 下载并安装适合你显卡的驱动。

---

## 🐍 二、安装 Anaconda 与 Python 虚拟环境

### 1. 安装 Anaconda（如未安装）

下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

### 2. 创建 Python 虚拟环境

```bash
conda create -n wafer_defect python=3.10
conda activate wafer_defect
```

---

## 📦 三、安装依赖项

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python matplotlib numpy
```

> ✅ 说明：`--extra-index-url` 是为了使用 **CUDA 11.8** 的 PyTorch 版本（适用于 NVIDIA GPU）。

---

## 📁 四、数据集结构与 YOLO 格式说明

数据集结构应如下所示（YOLOv8支持YOLO标注格式）：

```
wafer_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### `data.yaml` 文件内容如下：

```yaml
train: ./wafer_dataset/images/train
val: ./wafer_dataset/images/val
test: ./wafer_dataset/images/test

nc: 9
names: ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-Full', 'None', 'Random', 'Scratch']
```

---

## 🚀 五、使用 YOLOv8 训练晶圆缺陷检测模型

### 1. 下载 YOLOv8 官方预训练模型

```bash
# 例如下载 yolov8s.pt（中等大小）
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

### 2. 使用 Python API 训练模型

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8s.pt')

# 开始训练
results = model.train(
    data='wafer_dataset/data.yaml',  # 数据集配置文件
    epochs=100,                      # 训练轮数
    imgsz=640,                       # 图像大小
    batch=16,                        # 批次大小（根据显存调整）
    device=0,                        # 使用 GPU 0
    project='runs/wafer',            # 保存路径
    name='exp',                      # 实验名
    pretrained=True,
    optimizer='AdamW',               # 可选优化器
    lr0=0.001,                       # 学习率
    augment=True,                    # 数据增强
)
```

### 或使用命令行方式训练：

```bash
yolo task=detect mode=train model=yolov8s.pt data=wafer_dataset/data.yaml epochs=100 imgsz=640 batch=16 device=0
```

---

## 🔍 六、推理代码（模型检测）

### 1. 单张图片推理

```python
from ultralytics import YOLO
import cv2

model = YOLO('runs/wafer/exp/weights/best.pt')  # 加载训练好的模型

img_path = 'wafer_dataset/images/test/example.jpg'
results = model(img_path)

for r in results:
    im = r.plot()
    cv2.imshow('Defect Detection', im)
    cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. 视频或摄像头实时检测

```python
cap = cv2.VideoCapture(0)  # 或者传入视频路径

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for r in results:
        im = r.plot()
        cv2.imshow('Live Detection', im)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 七、模型评估（验证集测试）

```python
from ultralytics import YOLO

model = YOLO('runs/wafer/exp/weights/best.pt')  # 加载最佳模型
metrics = model.val(data='wafer_dataset/data.yaml')  # 验证集评估

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
```

---

## 📦 八、完整训练流程总结

| 步骤 | 命令 / 说明 |
|------|-------------|
| 创建环境 | `conda create -n wafer_defect python=3.10` |
| 安装依赖 | `pip install torch ultralytics opencv-python` |
| 准备数据 | 按照 YOLO 格式组织图像和标签 |
| 配置 data.yaml | 设置 `nc=9` 和类别名 |
| 开始训练 | `yolo task=detect mode=train ...` |
| 模型推理 | 使用 `model.predict()` |
| 模型评估 | 使用 `model.val()` 获取精度指标 |

以上文字及代码仅供参考学习使用。
