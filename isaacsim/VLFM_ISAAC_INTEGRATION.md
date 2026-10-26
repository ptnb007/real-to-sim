# VLFM 接入 Isaac Sim 5.1：中文操作说明

## 一、目前实现了什么

VLFM 和 Isaac Sim 分别运行在两个已经配置好的环境中：

- Windows 中的 Isaac Sim 负责加载房间和机器人，生成 `320×240` 的彩色图、深度图及小车位姿，并执行左右轮运动。
- WSL 中的 VLFM 通过 `localhost:12190` 接收观测，更新障碍地图、目标地图和语言价值地图，再返回线速度与角速度。
- Isaac Sim 根据 VLFM 返回的速度，通过差速控制器驱动 Jetbot。

数据流程如下：

```text
Isaac Sim 彩色图/深度图/位姿
            ↓
     WSL 中的 VLFM
            ↓
   线速度 + 角速度指令
            ↓
      Isaac Sim Jetbot
```

## 二、正常启动方法

### 第一步：启动 VLFM 服务

打开文件夹：

```text
C:\Users\woom\Desktop\woom\isaacsim
```

双击 `start_vlfm_server.bat`。这个窗口会：

1. 进入 `Ubuntu-24.04` WSL；
2. 激活 `vlfm` Conda 环境；
3. 启动 GroundingDINO、BLIP2-ITM 和 MobileSAM；
4. 启动 VLFM 与 Isaac Sim 之间的桥接服务。

第一次加载模型大约需要 60～90 秒。看到以下内容说明桥接服务已经就绪：

```text
Running on http://127.0.0.1:12190
```

不要关闭这个窗口。

### 第二步：启动 Isaac Sim 和 Jetbot

确认第一步的窗口仍在运行，然后双击 `run_vlfm_jetbot.bat`。程序会自动：

1. 打开 Isaac Sim 5.1；
2. 加载已经验证过的 Jetbot 场景；
3. 给 Jetbot 添加 RGB-D 相机；
4. 把相机图像和小车位姿发送给 VLFM；
5. 接收 VLFM 动作并驱动小车。

### 第三步：打开实时搜索界面

双击 `open_vlfm_dashboard.bat`，浏览器会打开：

```text
http://127.0.0.1:12190/dashboard
```

界面会实时显示 Isaac Sim 相机/目标检测画面、深度图、VLFM 障碍与探索地图、语言价值地图，以及当前搜索步数、速度和目标状态。

日志中出现以下信息表示连接成功：

```text
VLFM_SERVER_READY
VLFM_START goal='person'
VLFM_ACTION
```

当 VLFM 找到目标并到达目标附近时，会显示：

```text
VLFM_TARGET_REACHED
```

## 三、为什么小车开始时会旋转

这是正常现象，不是程序失控。

原版 VLFM 的真实机器人策略会先控制 Spot 的手部相机朝多个方向观察。Jetbot 没有能够独立旋转的手部相机，所以桥接程序改为让整个小车原地旋转，以获得不同方向的画面。初始化观察完成后，VLFM 才会开始前沿探索和目标导航。

## 四、修改搜索目标

用记事本打开：

```text
C:\Users\woom\Desktop\woom\isaacsim\start_vlfm_server.bat
```

找到 `--goal person`，将 `person` 换成需要搜索的英文目标，例如：

```text
--goal seated person
--goal lying person
--goal person wearing red clothes
```

目标描述建议使用英文，因为 GroundingDINO 和 BLIP2-ITM 的英文提示词更稳定。修改后需要重新启动 VLFM 服务。

## 五、当前场景的限制

当前验证场景只有 Jetbot 和地面，还没有真正的房间与人体。因此目前可以验证：

- Isaac Sim 输出 RGB-D 数据；
- VLFM 接收 Isaac Sim 的观测；
- VLFM 返回运动指令；
- Jetbot 执行运动指令。

要演示“机器人在房间中寻找被困人员”，还需要在 Isaac Sim 场景中加入：

1. 带碰撞属性的室内环境和地面；
2. 至少一个可见的人体模型；
3. 站立、坐姿或躺姿人体；
4. 障碍物以及可供小车通过的通道。

人体模型建议关闭网格碰撞，或只使用简单碰撞体，避免复杂人体网格阻塞小车。小车起点和人体位置必须位于相互连通的区域。

## 六、第一次使用时的文件同步

这些文件已经同步到本机的 `Ubuntu-24.04`，正常使用不需要重复执行。如果以后重新安装 WSL，可在 WSL 中执行：

```bash
cp /mnt/c/Users/woom/Desktop/woom/vlfm/vlfm/policy/isaacsim_policy.py \
  /home/woom/vlfm/vlfm/policy/
cp /mnt/c/Users/woom/Desktop/woom/vlfm/scripts/run_isaac_vlfm_server.py \
  /home/woom/vlfm/scripts/
cp /mnt/c/Users/woom/Desktop/woom/vlfm/scripts/launch_isaac_vlm_servers.sh \
  /home/woom/vlfm/scripts/
```

## 七、常见问题

### 1. 显示 `VLFM server is not reachable`

检查第一个窗口是否出现：

```text
Running on http://127.0.0.1:12190
```

如果没有，请等待模型加载完成；如果窗口已经报错，关闭两个窗口后重新执行启动步骤。

### 2. 小车轮子转动但车身不走

检查：

- Isaac Sim 是否正在播放物理仿真；
- 地面是否具有碰撞属性；
- 小车是否悬空或陷入地面；
- Jetbot 节点路径是否仍为 `/World/Jetbot`；
- 轮子关节是否为 `left_wheel_joint` 和 `right_wheel_joint`。

### 3. 小车一直原地转动

刚开始的多方向观察是正常的。如果长时间重复转动，可能是深度图无效、前沿地图没有可通行区域，或者场景只有空地而没有房间边界。需要检查相机视角、地面碰撞和室内障碍物。

### 4. 看到了人体，但没有识别

检查：

- 搜索词是否为英文；
- 人体是否位于相机画面内；
- 人体材质是否正常显示，而不是纯黑或透明；
- GroundingDINO 和 MobileSAM 服务是否成功启动；
- 人体与背景是否具有足够的视觉差异。

### 5. 显存不足

当前电脑为 8 GB 显存。建议只打开一个 Isaac Sim 场景，使用 `320×240` 相机分辨率和 Raytraced Lighting，不要同时启动其他深度学习程序。专用启动脚本不会启动当前流程不需要的 YOLOv7，以节省显存。

## 八、安全停止

按以下顺序停止：

1. 关闭 Isaac Sim，或者在 Isaac Sim 控制台按 `Ctrl+C`；
2. 在 VLFM 服务窗口按 `Ctrl+C`；
3. 如果还要释放 WSL 内存，可在 Windows PowerShell 中执行：

```powershell
wsl --shutdown
```

VLFM 判断已经到达目标时，桥接程序会先向 Jetbot 发送零速度指令，然后停止搜索。
# VLFM + InternNav iPlanner + Isaac Sim

## 当前控制链

```text
Isaac RGB-D → VLFM（语义目标/探索方向）
             → iPlanner（51 点局部避障轨迹）
             → 纯追踪控制（线速度/角速度）
             → 深度紧急避障（最后保护）
             → Jetbot 差速轮
```

VLFM 决定“去哪里”，InternNav/NavDP 中的 iPlanner 决定“眼前怎样绕过去”。
深度规则层不再承担主要规划，只在距离过近时减速、转向或急停。

## 启动顺序（3 个窗口）

1. 双击 `start_vlfm_server.bat`，等待 VLFM 服务启动。
2. 双击 `start_internnav_iplanner.bat`，看到 `Running on http://127.0.0.1:8888`。
3. 双击 `run_vlfm_jetbot.bat`，Isaac Sim 打开后开始搜索。

运行日志中出现以下内容即表示链路正常：

- `VLFM_SERVER_READY`：VLFM 已连接；
- `INTERNNAV_READY`：iPlanner 模型已加载；
- `INTERNNAV_PLAN`：收到局部轨迹及前视点；
- `LOCAL_AVOIDANCE`：紧急保护层发生介入。

如果 iPlanner 未启动，程序会打印 `INTERNNAV_FALLBACK` 并自动退回原 VLFM
控制，不会直接中止演示。将 `INTERNNAV_ENABLED=0` 可主动关闭 iPlanner。

