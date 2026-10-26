# Real-to-Sim：VLFM、Isaac Sim 与实验室场景重建

本仓库整理了本地 `woom` 工作区中已保存的代码、配置、操作文档和轻量验证报告，涵盖人体搜索、仿真桥接、机器人场景与真实实验室重建。它是多个实验阶段的源码归档，不代表所有模块已经集成为端到端系统。

## 模块导航

| 目录 | 内容与入口 |
| --- | --- |
| [vlfm](vlfm/) | VLFM 上游代码、本机复现适配、单人/多人搜索与 Isaac 服务；参见 [复现记录](vlfm/REPRODUCTION_STATUS.md)、[人体搜索](vlfm/PERSON_SEARCH.md) |
| [isaacsim](isaacsim/) | Jetbot、Galaxea R1 Pro 场景、导航桥接、实验室抓取演示；参见 [VLFM 接入说明](isaacsim/VLFM_ISAAC_INTEGRATION.md) |
| [real2sim_lab](real2sim_lab/) | SAM3 分割、房间网格构建、验证与打包；参见 [阶段说明](real2sim_lab/demo_segmentation/README.md) |
| [cruzr_isaac](cruzr_isaac/) | Isaac 刚体位姿到 Gaussian 绑定流的适配与测试；[说明](cruzr_isaac/README.md) |
| [gs_isaac](gs_isaac/) | GS-Playground/GaussianRenderer 与 Isaac 位姿、相机适配；[说明](gs_isaac/README.md) |
| [gaussian_splatting](gaussian_splatting/) | Nerfstudio/Splatfacto 训练、导出、遮罩与预览脚本；[说明](gaussian_splatting/README.md) |
| [lab_gs](lab_gs/) | 实验室局部 SfM 与 Gaussian 重建尝试；[质量与范围说明](lab_gs/README.md) |
| [astra_scene](astra_scene/) | 房间网格、USD 导出、机器人移动与碰撞验证脚本；[说明](astra_scene/README.md) |
| [mira_scene](mira_scene/) | 分割、深度、点云、RGB-D 网格与碰撞代理处理脚本 |

## 使用前须知

- 原工作环境涉及 Windows、WSL2、Isaac Sim 5.1、Blender 及多个独立 Python 环境；各模块依赖不同，不提供统一的一键安装承诺。
- 部分脚本保留原机绝对路径（如 `C:\Users\woom\Desktop\woom`、`D:\isaacsim-5.1.0`、WSL 挂载路径）。在其他机器或本仓库位置使用前，需要按模块说明调整路径，并准备缺失依赖及资产。
- `third_party/`、GroundingDINO、YOLOv7、本地依赖与运行环境未打包。请按模块 README、下载脚本及固定版本记录获取相应上游项目。
- 模型权重、照片/视频、场景与网格、训练检查点、采集数据、缓存及大型中间文件未上传。文档中指向这些文件的链接或路径可能仅在原工作区有效。
- JSON/TXT 验证结果是原工作区保存的历史记录，本次归档未重新执行 GPU 仿真、重建或导航实验。各模块已完成项、失败项和适用边界以其说明及报告为准。

## 来源与许可

`vlfm/` 包含 [Boston Dynamics AI Institute 的 VLFM](https://github.com/bdaiinstitute/vlfm) 及本地适配，保留原始 [MIT 许可证](vlfm/LICENSE) 和版权声明。其他依赖的代码、模型与数据遵循各自上游条款，本仓库未将它们重新打包，也未将整个仓库统一声明为 VLFM 的许可证。

详见 [上传范围](UPLOAD_SCOPE.md) 和 [源码清单](SOURCE_MANIFEST.json)。
