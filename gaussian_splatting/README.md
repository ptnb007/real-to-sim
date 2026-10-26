# 独立 3D Gaussian Splatting 复现

本目录先完成静态场景的独立闭环，之后才与 Isaac Sim 和机器人 Link 绑定。

## 固定方案

- 实现：Nerfstudio 1.1.5 / Splatfacto / gsplat
- 运行环境：WSL2 Ubuntu 24.04，独立 Conda 环境 `gsplat`
- 显卡：RTX 5060 8GB；使用默认 Splatfacto（官方说明约 6GB），不用 12GB 的 big 配置
- 首次验收数据：Nerfstudio 官方 `poster` 数据集
- 最终产物：训练配置、检查点、渲染结果、Gaussian PLY

## 验收顺序

1. `ns-train splatfacto` 能使用 CUDA 启动并完成短训练。
2. 网页 Viewer 能显示可交互高斯场景。
3. `ns-export gaussian-splat` 导出 `.ply`。
4. 再用自己的房间视频替换官方数据，经 COLMAP 求相机位姿。
5. 静态房间通过后，才进入机器人每个 Link 的 PLY 和位姿绑定。

## 与目标表格的对应关系

- `L2.ply + 环境 XML`：属于静态房间高斯与碰撞环境。
- `41 x Link PLY`：属于动态机器人外观，不是第一轮静态复现内容。
- `body_gaussians.json`：保存 Link 名称到 PLY 的映射。
- `cruzr_bind_manifest.json`：保存零位姿和毫米到米的坐标变换。
- 双相机和 `replay_*.py`：最后集成阶段完成。

## 当前状态

- Galaxea R1 Pro 已在 Isaac Sim 5.1 中完成导入和物理验收。
- Splatfacto / gsplat 的 CUDA 扩展已在 RTX 5060 上编译成功；官方数据 100 步验证通过。
- 自采实验室视频已抽取 66 帧，COLMAP 最佳模型注册 66/66 帧、得到 5022 个稀疏点，平均重投影误差约 0.93 px。
- 实验室场景已完成 7000 步训练，生成约 68 MB 的 Gaussian PLY 和三个训练相机预览。
- Windows 侧成果目录：`results/lab_20260929_134258/`。
- 当前模型没有真实尺度标定；视频中的移动人员会造成局部重影。后续进入 Isaac Sim 前需补尺度、静态碰撞几何和坐标对齐。

## 一键复现

在 Windows 中双击 `train_lab.bat`，或在 WSL 中运行：

```bash
GS_ITERATIONS=7000 bash /mnt/c/Users/woom/Desktop/woom/gaussian_splatting/train_lab.sh
```

脚本会自动选择注册帧数最多的 COLMAP 模型，并依次完成训练、PLY 导出和预览渲染。
