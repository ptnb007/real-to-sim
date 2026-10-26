# 酒店房间 Mesh

本模型对应上传的 GS-Playground 酒店房间四视角拼图。
从原始导航场景 3D Gaussian Splatting 资产渲染 108 个带标定的 RGB/期望深度视角，经过透明度过滤和 TSDF 融合提取三角网格。
不是仅凭拼图猜测的模型，也不是 SAM3D 生成结果。

顶点：246516；三角面：472219。

- room_mesh.ply：带顶点颜色的三角网格，注意不是 Gaussian PLY。
- room_mesh.obj：通用三角网格；不同软件对 OBJ 顶点颜色支持不同，颜色优先用 PLY。
- room_mesh.usda：Z 轴朝上，已添加静态三角网格碰撞。Isaac Sim 中可打开；尚未做仿真碰撞验证。
- mesh_preview.png：实际三角网格的四视角射线投射画面，没有混入 Gaussian 渲染。
- report.json：提取参数、数量及几何检查结果。
- build_room_mesh.py：提取脚本；依赖原仓库场景、GaussianRenderer、匹配版本 gsplat、PyTorch、Open3D 和 USD Python。

限制：保留源 GS 坐标与尺度，未通过实物尺寸标定；此模型是静态表面，遮挡背面可能缺失，未独立拆分床、桌椅等物体。TSDF 深度来自外观优化的 GS，因此表面可能起伏、孔洞和局部误差。尚不能当作完成验证的机器人仿真场景。

原始资产：third_party/gs_playground/demo/navigation/nav_scene_1/3dgs/point_cloud.ply。
