# GS-Real2Sim 官方 Demo：SAM3 分割验证

本次已使用本机登录凭据验证 SAM3、SAM-3D-Objects 和 AnySplat 文件访问均为 HTTP 200。没有在代码或结果包中保存 Token。

已实际运行 SAM3 官方图像模型，对 GS-Real2Sim 自带 example/example.png 分别使用 pot、bottle、towel 提示词，三个物体各输出一个掩码。结果见 report.json、sam3_demo_masks.png 与 masks/。

这只完成了二维分割，不是单独三维模型。背景修复、AnySplat 背景重建、SAM3D 物体 Mesh/GS 生成及 Isaac 物理导入尚未完成。

## 实际环境

- RTX 5060 8GB；Isaac 自带 Python 3.11 / PyTorch 2.7.0+cu128。
- 模型权重 sam3.pt 已下载到本机（约 3.45GB），不包含在此包内。
- GPU 分割使用 BF16 autocast，官方默认 1008 处理分辨率；未修改神经网络和权重。
- Windows Triton DLL 无法加载，使用 USE_PERFLIB=0 关闭上游可选优化，并将其 EDT 辅助函数接到 OpenCV CPU 实现。该兼容函数用于导入/交互跟踪依赖，不以假模型替代 SAM3。当前只验证单张图像分割。

## 固定源码版本

- GS-Real2Sim: 927017e91087ed7e74036c68dca17ceb28ede1fb
- SAM3: f1647e8d0c1d280991f4d1b7d974c2704537fe66
- iopath: 3762ffbcce3cdc7038d655f7c1955a7441a441db

脚本为 run_demo_segmentation.py、sam3_windows_compat.py。当前程序路径依赖工作区中第三方源码和独立 deps 目录；此包是结果与运行代码记录，并非完整离线安装包。
