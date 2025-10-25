# Contour Deep: Deep Learning for Point Cloud Place Recognition

基于多层BEV表示的深度学习点云场景识别方法

## 项目结构
```
contour_deep/
├── configs/          # 配置文件
├── data/            # 数据处理模块
├── models/          # 神经网络模型
├── training/        # 训练脚本
├── evaluation/      # 评估脚本
├── utils/           # 工具函数
├── scripts/         # 预处理和测试脚本
├── checkpoints/     # 模型权重保存
├── logs/            # 训练日志
└── results/         # 评估结果
```

## 环境配置
```bash
# 创建conda环境（如果还没有）
conda create -n contour_deep python=3.9
conda activate contour_deep

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 数据预处理
```bash
python scripts/preprocess_bev.py --config configs/config_base.yaml
```

### 2. 训练方向1（检索特征网络）
```bash
python training/train_retrieval.py --config configs/config_retrieval.yaml
```

### 3. 训练方向2（BCI匹配网络）
```bash
python training/train_bci.py --config configs/config_bci_matching.yaml
```

### 4. 端到端评估
```bash
python evaluation/eval_pipeline.py --config configs/config_base.yaml
```

## 引用

如果使用本项目，请引用：
```
@inproceedings{contour_deep_2024,
  title={Contour Deep: Learning-based Place Recognition with Multi-layer BEV},
  author={Your Name},
  year={2024}
}
```

## 许可证

MIT License
