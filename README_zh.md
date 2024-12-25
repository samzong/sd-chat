# SD Chat - AI 创作平台

[English](README.md) | [简体中文](README_zh.md)

一个基于 Stable Diffusion 的 AI 创作平台，支持图像生成和视频生成功能。本项目集成了多个先进的 AI 模型，提供简单易用的 Web 界面，让用户可以轻松创作高质量的 AI 图像和视频。

## Roadmap

### 已完成功能 ✅

#### 1. 文生图（Text-to-Image）
- [x] Stable Diffusion XL Base 1.0
- [x] Stable Diffusion 1.5
- [x] Realistic Vision V5.1
- [x] Dreamshaper V8
- [x] 支持自定义提示词
- [x] 支持参数调节
- [x] 支持负面提示词

#### 2. 文生视频（Text-to-Video）
- [x] Stable Video Diffusion
- [x] 支持自定义提示词
- [x] 支持视频参数调节
- [x] 支持负面提示词

#### 3. 图生视频（Image-to-Video）
- [x] Stable Video Diffusion
- [x] 支持自定义起始图片
- [x] 支持视频参数调节
- [x] 支持运动强度控制

### 开发中功能 🚧

#### 4. 图像编辑（Image Editing）
- [ ] ControlNet 支持
- [ ] 图像修复（Inpainting）
- [ ] 图像扩展（Outpainting）
- [ ] 提示词编辑（Prompt-based Editing）

#### 5. 视频编辑（Video Editing）
- [ ] 视频修复
- [ ] 视频风格转换
- [ ] 视频帧插值
- [ ] 视频分辨率提升

## 功能特点

### 1. 图像生成
- 支持多个高质量模型：
  - Stable Diffusion XL
  - Stable Diffusion 1.5
  - Realistic Vision
  - Dreamshaper
- 自定义生成参数
- 高质量输出
- 批量生成支持

### 2. 视频生成
- 基于 Stable Video Diffusion
- 支持图片到视频转换
- 支持文本到视频生成
- 可调节视频参数

### 3. 界面特性
- 直观的 Web 界面
- 实时预览
- 参数调节控制
- 示例提示词

## 快速开始

### 环境要求
- MacOS 操作系统（Apple Silicon）
- Python 3.10+
- 32GB 内存（推荐）
- 磁盘空间：至少 20GB（用于模型存储）

### 安装步骤

1. 安装 Conda（如已安装可跳过）：
```bash
# 下载 Miniforge3（适用于 Apple Silicon）
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh
```

2. 创建并激活环境：
```bash
# 创建 Python 3.10 环境
conda create -n diffusers python=3.10
conda activate diffusers
```

3. 安装 PyTorch（针对 MPS 加速）：
```bash
# 安装支持 MPS 的 PyTorch
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

4. 克隆项目：
```bash
git clone https://github.com/samzong/sd-chat.git
cd sd-chat
```

5. 安装依赖：
```bash
pip install -r requirements.txt
```

### 首次运行

1. 启动服务：
```bash
python run.py
```
首次运行时会自动下载所需模型，这可能需要一些时间，具体取决于网络状况。

2. 访问界面：
在浏览器中打开 http://localhost:7860

### 目录结构
```
sd-chat/
├── run.py          # 统一启动脚本（推荐使用）
├── api.py          # 图像生成服务
├── video_api.py    # 视频生成服务
├── web_ui.py       # Web 界面
└── requirements.txt # 项目依赖
```

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。 