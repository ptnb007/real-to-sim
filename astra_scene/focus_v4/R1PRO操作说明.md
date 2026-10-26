# focus_v4 + Galaxea R1 Pro 操作说明

## 1. 打开最新组合场景

双击 `..\open_focus_v4_r1pro_isaac.bat`。

组合场景文件是 `focused_zone_with_r1pro.usda`。机器人出生点位于桌子之间的开放区域，位置为 `(0.25, -0.55, 0)`。

## 2. 为什么直接设置轮速时可能不动

R1 Pro 的 URDF 导入后，连续驱动轮默认仍带有位置驱动刚度。仅设置速度目标时，位置驱动会把轮子锁在零位。控制前需要把三个轮关节的刚度 `Kp` 设为 0，或者直接使用轮关节力矩控制。

三个转向关节：

- `steer_motor_joint1`
- `steer_motor_joint2`
- `steer_motor_joint3`

三个驱动轮关节：

- `wheel_motor_joint1`
- `wheel_motor_joint2`
- `wheel_motor_joint3`

## 3. 运行已经验证的移动测试

安全运动演示请双击 `..\open_r1pro_safe_motion_demo.bat`。它会加载 Galaxea
官方 R1 Pro 仿真资产，采用“转向关节位置控制 + 车轮速度控制”，先稳定上身姿态，
再缓慢加速、直行和减速。请不要再使用三个轮子直接施加相同大扭矩的旧脚本。

只查看最新组合场景可双击 `..\open_focus_v4_official_r1pro_isaac.bat`。

无界面自动验证可双击 `..\run_r1pro_safe_motion_test.bat`。脚本会：

1. 打开最新 focus_v4 场景；
2. 将三个转向轮设为零角度；
3. 给三个驱动轮施加力矩；
4. 推进 30 个物理帧；
5. 把移动前后坐标写入 `r1pro_drive_verification.json`。

本机验证结果：机器人移动约 `0.213 m`，同时发生了转向。该测试用于证明底盘和轮地接触有效，并不是已经标定好的直线控制器。

## 4. 生成 Domain Randomization 场景

在此项目目录运行：

```powershell
python astra_scene\generate_focus_v4_randomization.py
```

输出位于 `randomized` 文件夹，包括 12 个可重复的 USD 变体和一个 `manifest.json`。当前随机化内容：

- 机器人初始 XY 位置；
- 机器人偏航角；
- 环境光强度和颜色；
- 主光强度；
- 相机焦距。

这些 USD 是场景变体，不是已经渲染好的训练图片。下一阶段可使用 Isaac Sim Replicator 为每个变体输出 RGB、深度图、语义分割、实例分割和标注 JSON。

## 5. 控制建议

- 基础验证先采用轮关节力矩控制；
- 正式导航应实现 R1 Pro 三舵轮运动学，把 `(vx, vy, wz)` 转换为三个转向角和三个轮速；
- 机器人运动前必须按 Play 启动物理仿真；
- 8 GB 显存下建议保持 960×540，并一次只加载一个随机化场景。
