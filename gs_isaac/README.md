# GS-Playground → Isaac Sim 最小适配

## 已落实的工作

本项目围绕实际 GS-Playground 技术路线，不依赖参考 PPT 中的 CRUZR 工程。

- 下载并阅读了官方 GS-Playground 与 GaussianRenderer 源码，未修改上游文件。
- 加载官方导航场景的 265,778 个 Gaussian 到 RTX 5060 GPU。
- 实现 Isaac Sim 刚体位姿 → 官方 GSRenderer 的动态更新适配。
- 实现 USD 相机世界矩阵、垂直 FOV → 官方 render_batch 的参数转换。
- 在 Isaac Sim 5.1 内实际运行自由落体及接触，连续同步 120 帧；位置误差小于 4e-8 米，背景保持不动。
- 已解决渲染内核问题：使用匹配 Python 3.11 / Torch 2.7 / CUDA 12.8 的社区预编译 gsplat，独立安装并校验 SHA-256。
- 官方导航场景已输出四个 960×720 视角，见 results/GS_Playground_Views.png。此图为静态 GS 渲染，不代表已完成机器人导航。

最终状态以 results/integration.json 和 results/renderer_probe.json 为准。测试物体是合成 Gaussian 立方体，不代表机器人重建、导航或抓取完成。

## 源码接口与结论

| 源文件 | 实际入口 | 用途 |
|---|---|---|
| GaussianRenderer/src/gaussian_renderer/core/gs_renderer.py | GSRenderer(models_dict) | 加载 PLY，将 Gaussian 上传 CUDA |
| 同上 | set_objects_mapping(objects_info) | 按稳定顺序指定 Gaussian 与动态物体的对应关系 |
| 同上 | update_gaussian_properties(pos, quat, scalar_first=True) | 输入世界位置和 wxyz 四元数，更新 Gaussian 中心及朝向 |
| 同上 | render_batch(cam_pos, cam_xmat, height, width, fovy_arr, y_up=True) | 输入相机参数，返回 RGB 和深度通道 |
| GaussianRenderer/src/gaussian_renderer/gs_renderer_motrixsim.py | update_gaussians(mx_data) | 读取 Motrix link poses，按 xyzw 调用通用更新入口 |
| GaussianRenderer/src/gaussian_renderer/gs_renderer_isaacgym.py | GSRendererIsaacGym | 已有 Isaac Gym 包装；不是 Isaac Sim 适配 |
| GaussianRenderer/src/gaussian_renderer/batch_splat.py | BatchSplatRenderer | 仿真器无关的批量入口，后续多环境可参考 |
| gs_playground/demo/live_demo/replay.py | BatchSplatConfig / MtxBatchSplatRenderer | 官方批量渲染用法 |
| gs_playground/demo/navigation/robot_locomotion.py | collect_robot_gaussians / apply_gaussian_links | 机器人 PLY 名称与物理 Link 名称的对应方式 |

Motrix 包装中的一条注释写 wxyz，但实际调用 scalar_first=False；适配必须以代码行为为准。本项目直接使用 Isaac get_world_pose() 的 wxyz，显式传 scalar_first=True。

## 本次适配的数据约定

- 长度统一为米，世界坐标采用 Z-up。
- 动态 PLY 已在对应刚体的局部坐标系中；如资产另有尺度、偏移，需要预处理后再绑定。
- bodies 的顺序同时决定 Gaussian 映射和位姿输入顺序。
- USD 相机使用 -Z 前向、+Y 向上；上游 y_up=True 翻转相机 Y/Z 列，再求逆得到 view matrix。这里不是把世界轴改为 Y-up。
- 垂直 FOV = 2 atan(verticalAperture / (2 focalLength))，传入角度值。
- 当前接口只支持中心主点、方形像素；相机 aperture 宽高比必须与输出分辨率相同。不支持任意畸变或偏移主点。
- 上游 render_mode 是 RGB+D，其 D 是累积深度通道，不能直接当作已验证的度量 RGB-D 传感器输出。
- 当前沿用上游 Gaussian 朝向更新机制；高阶 SH 的方向一致性没有单独验证。
- 为绕过 Windows 上 torch.compile 的额外编译依赖，bootstrap.py 禁用了 torch.compile；没有替换上游变换函数，也没有用模拟渲染器代替 gsplat。

## 在本机运行

已安装 Isaac Sim：D:/isaacsim-5.1.0，Python 3.11，PyTorch 2.7.0+cu128，RTX 5060 8GB。

1. 首次用系统 Python 执行 `python gs_isaac/fetch_deps.py`。依赖仅解压到本目录，不改 Isaac 安装。下载文件校验 PyPI SHA-256。
2. 执行 `gs_isaac/run_probe.bat`：读取官方导航场景并尝试渲染。
3. 执行 `gs_isaac/run_integration.bat`：启动 Isaac，运行物理与 Gaussian 同步，再尝试渲染。
4. 执行 `gs_isaac/run_views.bat`：输出四个 960×720 视角及拼图。

压缩包内已包含校验过的预编译 gsplat；如需重新下载，运行 `python gs_isaac/fetch_prebuilt.py`。其来源、版本和摘要见 prebuilt_source.json；该二进制由社区构建，并非 gsplat 官方发行的 Windows wheel。

脚本通过 uv 启动 Isaac Python。退出码 0 = 当前脚本所有检查通过；1 = 物理或适配失败；2 = 物理/加载可通过，但渲染不可用。

results/ 下保留机器可读报告、120 帧位姿及测试 USD。日志含原始失败堆栈，便于继续定位。

## 已解决的阻塞与下一步

此前的 gsplat 纯 Python wheel 因缺少 CUDA Toolkit 无法启用内核。现通过本地 prebuilt 目录中的匹配扩展解决，不需要修改 Isaac 安装或安装系统级工具链。

静态渲染与集成结果以当前 results 下的 JSON 和 PNG 为准。后续进入真实 Mesh 碰撞场景的导入与对齐验证。若转 Linux，需保留 Isaac Python ABI 与 gsplat 依赖的一致性；官方预览项目自身固定 Python 3.10，因此不要直接覆盖 Isaac 3.11 环境。

接着验证官方场景 Mesh 与 GS 的空间对齐、碰撞及移动相机遮挡，最后接移动机器人和 FAR Planner。官方 Navigation Demo 是 locomotion/场景演示，不能当成 FAR Planner 已接入。

当前不声称已完成：完整官方 Demo 复现、导航 Mesh 导入 Isaac、GS/Mesh 遮挡验证、机器人迁移、FAR Planner 或操作任务。

## 固定的上游版本

- https://github.com/discoverse-dev/GaussianRenderer ：577efdb50c3891f46db3971dc0b7c4129146dd16
- https://github.com/discoverse-dev/gs_playground ：33127caa0e05b59c341628aee266ce7418dbe184
- https://arxiv.org/abs/2604.25459

压缩包包含必要上游渲染源码、许可证和官方导航 GS/Mesh 样例；未打包整套上游大资产、Isaac 安装或下载依赖。解压后保留 gs_isaac 与 third_party 的同级关系。
