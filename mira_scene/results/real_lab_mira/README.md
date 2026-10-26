# 真实实验室 Mira 重建（进行中）

本目录与旧 v4 Blender/Isaac 场景、v4 合成图测试完全分离。

## 输入
- purple_table：用户视频的真实 frame_00.jpg，包含前桌、后桌、紫色挡板。
- white_divider：用户真实照片 5473f5c5d4281be7c15cc84b828c47d6.jpg。
- purple_divider：同一真实视频帧，SAM3 框选辅助分割（候选 2，经视觉核对）。

## 已验证
- SAM3 真实照片分割：桌子、白色挡板；紫色挡板另用交互提示。
- purple_table 与 white_divider：官方 Mira CCM 推理 30 步完成，退出码 0。
- 上述两幅图：MoGe 深度估计完成，退出码 0；深度未经真实尺寸标定。
- 输出包含 CCM 坐标图、canonical_pcd、voxel_coords；它们是中间结果，不是带纹理场景。

## 尚未完成
- SAM3D 带纹理网格、Mira 场景配准/优化、最终 GLB/USD。
- 两张不同视角输入尚未统一为一个场景，不能直接拼接其坐标。
- 前桌被图像边界截断，桌面物品和机器人遮挡部分缺少观测。
- 人物/机器人不作为待生成物体；去掉实例并不等于恢复被其遮挡的真实背景。

## 本轮新增：RGB-D 彩色点云
- `rgbd_pointcloud/purple_table.ply`：真实视频帧中的桌子/紫色挡板区域，106,228 点。
- `rgbd_pointcloud/white_divider.ply`：真实照片中的白色折叠挡板区域，57,460 点。
- `rgbd_pointcloud/index.html`：本地可旋转、缩放的查看器；两幅图按独立相机视角切换，不代表已做多视角配准。
- `rgbd_pointcloud/pointcloud_preview.png`：静态预览。
- PLY 来自真实 RGB 与 MoGe 相对深度融合，深度没有米制标定；这是点云，不是闭合网格，也不能替代完整 Mira/SAM3D 重建。

## 硬件与环境
- RTX 5060 8GB；SAM3D 官方 setup.md 要求至少 32GB 显存。
- CCM 实测可运行；SAM3D 尚在依赖兼容验证，不宣称已验证显存溢出。
- CUDA 12.8 编译器在 gsplat 环境内；不在默认 PATH，不是未安装。
- PyTorch3D 已在 CCM 环境源码编译成功；SAM3D 网格阶段仍未能初始化，当前阻塞在 SAM3D 需要的 `utils3d.numpy.depth_edge` 与已装 MoGe 工具包 API 不兼容。未生成任何物体 GLB。
- 官方 SAM3D 设置要求至少 32GB 显存；当前 RTX 5060 只有 8GB。阶段二省略未使用的阶段一网络是试验性低显存尝试，不代表已通过支持配置。

最终验收需有真实照片→Mira 带纹理网格→布局对齐→渲染对比，不能用旧 v4 场景替代。
