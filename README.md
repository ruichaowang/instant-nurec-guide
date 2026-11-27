# 用 Instant-NuRec 从多路行车图像生成自由视角

基于预训练的 [NVIDIA/instant-nurec](https://github.com/NVIDIA/instant-nurec)（`pa-multiview`）与可选单步神经修复模型 `nvidia/difix_ref`，在不训练生成模型的前提下，将官方片段级离线重建改造成**五路因果短窗更新**与**浏览器自由视角查看器**。

## 1. 项目目的与定位

上游官方入口 `run_inference.py` 面向片段级离线重建与 `.ply` + sky 文件导出（默认使用 3 路相机 × 18 个时刻），缺少“按时间因果推进 + 内存复用高斯交互换机位 + 同步五路实拍对照 + 可选单步修复”的轻量可运行参考实现。

本仓库用精简的代码（[render_view.py](render_view.py)、[viewer.py](viewer.py)、[viewer.html](viewer.html)）提供一套可直接复跑的工程参考：

- **定位与目标**：验证五路车载相机的因果短窗 3DGS 重建、序列图片导出与本地浏览器交互换机位能力。
- **边界与非目标**：本仓库**不包含任何数据集或模型权重**；面向单卡 GPU 原型验证与效果对照，不等同于车端实时量产方案。

## 2. 核心原理

### 2.1 数据流概览

```text
5 路 NCore 曝光 + 历史车体位姿 (<= t)
  └─> 因果短窗选帧 (select_frames: 每路 4 帧，共 20 张 504x280)
        └─> 窗口局部参考系归一化 (Scene.prepare / load_batch)
              └─> pa-multiview 内存前馈重建 (Scene.reconstruct: 3DGS + Sky)
                    └─> 虚拟相机射线与位姿刚体变换 (render -> 1024x576 RGB)
                          ├─> [默认] 输出原始 GS + 天空画面
                          └─> [可选 --difix] 同进程单步修复 (enhance)
                                └─> PNG 序列导出 / HTTP 浏览器查看器
```

### 2.2 关键机制（与源码对应）

1. **五路因果短窗选帧（`select_frames`）**：使用前广角（`camera_front_wide_120fov`）、左前侧（`camera_cross_left_120fov`）、右前侧（`camera_cross_right_120fov`）、左后（`camera_rear_left_70fov`）、右后（`camera_rear_right_70fov`）共 5 路相机。给定查询时刻 `t`，仅读取 `<= t` 的历史曝光与车体位姿（严格因果，不使用未来帧）。在过去 2 秒窗口内按车辆行驶弧长对齐官方 18 帧采样比例（`[0, 6, 11, 17]`）每路选 4 帧，共 20 张 `504×280` 图像；若车辆静止或位移过小导致去重后不足 4 帧，则自动回退为窗口内按时间下标等距取 4 帧。
2. **窗口局部坐标归一化（`Scene.prepare` / `load_batch`）**：以本窗最早一帧前广角相机的世界位姿作为参考原点，将 5 路输入与目标渲染帧统一变换到局部坐标系，避免车辆长距离行驶后全局坐标超出模型位姿编码分布。
3. **纯内存无落盘重建（`Scene.reconstruct`）**：直接调用 `pa-multiview` 模型的前馈重建与导出预处理接口，在 GPU 内存中生成当前时刻的局部 3D 高斯与天空，省去 `.ply` 文件写盘与重新解析。
4. **保留标定镜头模型的虚拟相机渲染（`render`）**：在车体坐标系中按环绕角（`yaw`）、俯角（`pitch`）、离地高度（`height`）和车后水平距离（`rear`）构造虚拟相机位姿，对 `1024×576` 前视标定射线与位姿做刚体变换后完成 GS + 天空合成，并在渲染结束后在 `finally` 块中恢复原张量。因此**暂停后切换机位可直接复用内存中的当前高斯，无需重新重建**。
5. **单线程后台预取与可选同进程 Difix（`enhance`）**：时间推进时使用单个后台工作线程（`ThreadPoolExecutor(max_workers=1)`）提前组装下一窗输入；开启 `--difix` 时，在同一 GPU 进程内将渲染 RGB 与当前时刻真实前广角图像送入 `nvidia/difix_ref` 做单步（`timestep=199`，BF16）2D 修复，不修改底层 3D 高斯场景。

| 当用户做什么 | 服务端执行什么 |
| --- | --- |
| 播放到新时刻 | 按五路各自的曝光时间，从当前及历史图像中各选 4 张，共 20 张。将相机位姿变换到本窗参考系，调用重建核心，得到这一时刻的局部高斯和天空。此分享版用一个后台工作线程提前组装下一窗的五路输入。 |
| 暂停后改变机位 | 复用当前时刻的场景，只更新虚拟相机的位置、朝向和射线，再渲染一次。机位可用环绕角、俯角、离地高度和车后水平距离描述。 |
| 开启 Difix | 将虚拟视角渲染的 RGB 和当前时刻的真实前视参考图作为输入，在同一 GPU 进程中做单步修复，得到当前输出帧。修复发生在二维画面上，高斯场景本身保持原样。 |

工程上只在内存中保留当前时刻的场景：时间推进时重新生成局部高斯，机位变化时复用当前高斯，跳转时间轴时重新准备输入窗。

## 3. 当前效果与能力边界

### 3.1 视觉效果与已知局限

基于公开 NCore 驾驶片段（输出分辨率 `1024×576`）的观察结果：

- **低位追车（默认 `rear=4.5 m, height=2.4 m, pitch=12°`）**：道路、沿街建筑与周围车辆轮廓可辨，支持连续时间窗推进与暂停后绕车交互；查看器下方同步显示同一时刻五路实拍缩略图便于对照。
- **高位视角（例如 `rear=4 m, height=5 m, pitch=45°`）的已知不足**：实拍未覆盖的遮挡区（如围栏后方、墙后、屋顶等）仍会出现灰面或空洞；细杆、围栏等薄结构边缘容易模糊；部分高速或局部遮挡的动态目标还原不全。
- **Difix 修复表现**：在与前视参考图存在视野重叠的区域，能够改善车道线与边缘清晰度、减轻部分渲染伪影；由于修复仅在二维图像空间进行且高位视角缺少同机位实拍真值，强遮挡区可能出现“观感更自然但内容未必真实”的纹理补全，无法据此验证遮挡区真实几何。

### 3.2 验证状态与性能参考

- **本分享版仓库已直接验证项**（NVIDIA A100 80 GB 环境）：
  - 连续两个时刻分别导出了原始 GS 图与 Difix 修复图，四张输出分辨率均为 `1024×576`。
  - 查看器 HTTP 接口通过三项核验：转动机位后画面发生改变、转回原机位后像素保持一致、推进到下一时刻后底部五路实拍对照同步更新。
- **同模型、同短窗配置的单卡参考性能**（NVIDIA A100-SXM4-80GB，5 路 × 4 时刻，输出 `1024×576`，基于 100 窗串行/独立对照测得的量级参考，非本分享版单后台线程代码的独立压测值）：

| 链路 / 场景 | 耗时量级参考 | PyTorch 显存参考 | 说明 |
| --- | --- | --- | --- |
| 固定时刻交互换机位（复用内存 3DGS） | 平均约 **130 ms/次** | 复用当前场景 | 含浏览器请求、HTTP 传输与 JPEG 编解码；底层光栅化本身约 **7 ms** |
| 单窗完整更新（原始 GS + 天空，无缓存串行） | 中位数约 **1.5 s/帧** | 分配峰值约 **2.9 GiB** / 保留约 **4.8 GiB** | 输入准备约 1.1 s + 重建约 0.28 s + 机位准备与光栅化约 0.10 s |
| 叠加同进程单步 Difix（BF16，1 步） | 中位数约 **1.8–2.4 s/帧** | 分配峰值约 **7.6 GiB** / 保留约 **10.1 GiB** | 其中纯 Difix 修复阶段约 **0.5–0.85 s/帧** |

> [!NOTE]
> **性能口径说明**：本分享版采用**单个后台线程预取下一窗**（不包含多路曝光显存缓存与五路独立读取线程），可在一定程度上掩盖下一窗输入准备耗时，但此前五路独立读取线程的优化数字不能直接套用；实际端到端耗时受 GPU 型号、CPU 图像解码吞吐以及 PyTorch/驱动版本影响，请以本机实测为准。整体仍处于单卡云端准实时原型量级（连续更新约 0.5–1 FPS），尚未达到车端实时显示要求。

## 4. 准备输入与官方基线验证

### 4.1 准备输入

| 输入 | 怎么准备 |
| --- | --- |
| 多路 RGB 图像 | 五路方案使用前广角、左侧、右侧、左后和右后。保留每路相机的身份和逐帧曝光时间戳，方便按同一车辆时刻选图。 |
| 相机标定 | 保留原始分辨率、投影模型及畸变参数、相机相对车体的外参。图像缩放或裁切后，投影参数也要对应更新。 |
| 车辆运动 | 准备与图像使用同一时间基准的车体位姿序列，明确坐标轴、变换方向和长度单位。滚动快门相机还需保留读出时序信息。 |
| 权重 | 重建使用 Instant-NuRec 的 `pa-multiview`。需要比较神经修复效果时，再准备 `nvidia/difix_ref`。 |

使用自采数据时，先选一小段行驶片段，检查同一处路标在相邻时刻、相邻相机里的投影能否对齐。官方命令读取 NCore 格式的序列描述及其引用文件；使用其他格式时，需要转换数据或实现读取适配器，单独提供若干 MP4 视频缺乏精确时间基准与几何标定信息。

### 4.2 先验证官方重建

**获取官方样例。** NVIDIA 在 GitHub README 中指定了一个示例片段，原始数据由 [Hugging Face 上的 NCore 数据集](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore) 提供。使用自己的账号接受数据集条款后，可在终端只下载该片段（官方标注约 2 GB）。

```bash
hf auth login
clip_id=000da9de-0ee5-465a-9a2d-e7e91d3016bb
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NCore --repo-type dataset \
  --include "clips/$clip_id/*" --local-dir ./demo_clip
```

在上游仓库根目录准备好 GPU 环境和权重后，设 `seq="./demo_clip/clips/$clip_id/pai_$clip_id.json"`。先用官方入口验证前视基线与多相机版本：

```bash
python run_inference.py --model pa-front --ncore-path "$seq" --output-dir ./demo_output --merge

python run_inference.py \
  --model pa-multiview \
  --ncore-path "$seq" \
  --output-dir ./output \
  --merge
```

官方默认多相机配置使用 3 路相机、每路 18 个时刻。要显式指定 5 路，可按官方支持的相机顺序重复传入 `--camera-id`。输出目录中的 PLY 保存合并后的高斯，配套的 sky 文件保存天空信息；只在通用查看器中打开 PLY 时，天空需要另外加载。本仓库基于上游 commit [`f2ecafcdf28f2dc8f2cc17447a772571cde558be`](https://github.com/NVIDIA/instant-nurec/blob/f2ecafcdf28f2dc8f2cc17447a772571cde558be/README.md) 验证通过。

## 5. 运行本仓库（导出序列与交互查看器）

在 Linux NVIDIA GPU 上按上游说明完成安装，包括 `uv sync --extra render`。使用已激活的上游 Python 环境，把上游仓库根目录加入 `PYTHONPATH`，然后在**本仓库根目录**运行（将以下路径替换为实际路径）：

```bash
export PYTHONPATH="/path/to/instant-nurec:${PYTHONPATH:-}"
seq=/path/to/sequence.json
ckpt=/path/to/instant_nurec_pa_multiview_1.1.0.pth

# 从共同覆盖区的第 2 秒开始，每 0.2 秒输出一张，含天空。
python render_view.py --sequence "$seq" --checkpoint "$ckpt" \
  --start 2 --frames 10 --step 0.2 --yaw 0 --pitch 12 --height 2.4 --rear 4.5 --output output

# 用同一段数据启动本地交互查看器。远程使用时，通过 SSH 转发此端口。
python render_view.py --sequence "$seq" --checkpoint "$ckpt" --start 2 --frames 10 --serve 8767
```

本机浏览器打开 `http://127.0.0.1:8767/`。导出模式会生成 `gs-00000.png` 等文件；按场景时间合成为视频可运行：

```bash
ffmpeg -framerate 5 -i output/gs-%05d.png -c:v libx264 -pix_fmt yuv420p chase.mp4
```

其中 `5` FPS 对应上述 `0.2` 秒采样间隔。

**可选 Difix 修复。** 本仓库仅包含 Difix 调用胶水代码（[render_view.py](render_view.py) 中的 `load_difix` / `enhance` 与 [viewer.py](viewer.py) / [viewer.html](viewer.html) 的网页开关），**不包含** [Difix3D 官方源码](https://github.com/nv-tlabs/Difix3D) 与 `nvidia/difix_ref` 模型权重。

准备完整的 `nvidia/difix_ref` 本地快照（含 `vae`、`unet` 模块），并将 `Difix3D` 的 `src` 加入 `PYTHONPATH` 以便导入 `pipeline_difix`。在已验证的组合环境中，Difix 侧依赖版本为 `diffusers==0.25.1`、`transformers==4.38.0`、`accelerate==0.28.0`、`peft==0.9.0`、`huggingface_hub==0.25.1`；由于上游 NuRec 对 `huggingface_hub` 的锁定版本不同，建议为组合实验单独准备环境。

```bash
# 1. 拉取 Difix3D 源码（提供 pipeline_difix）并下载 nvidia/difix_ref 本地快照
git clone https://github.com/nv-tlabs/Difix3D.git /path/to/Difix3D
hf download nvidia/difix_ref --local-dir /path/to/difix_ref

# 2. 同时将 instant-nurec 与 Difix3D/src 加入 PYTHONPATH
export PYTHONPATH="/path/to/instant-nurec:/path/to/Difix3D/src:${PYTHONPATH:-}"
difix_dir=/path/to/difix_ref

# 3. 导出模式：每个时刻同时保存 gs-00000.png 与 difix-00000.png
python render_view.py --sequence "$seq" --checkpoint "$ckpt" --difix "$difix_dir" \
  --start 2 --frames 10 --step 0.2 --yaw 0 --pitch 12 --height 2.4 --rear 4.5 --output output

# 4. 交互查看器模式：启动后可在网页勾选/取消 Difix 开关
python render_view.py --sequence "$seq" --checkpoint "$ckpt" --difix "$difix_dir" \
  --start 2 --frames 10 --serve 8767
```

### 命令行参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sequence` | 必填 | NCore 序列 JSON 文件路径 |
| `--checkpoint` | 必填 | `pa-multiview` 模型权重路径（`.pth` / `.pt`） |
| `--output` | 无 | PNG 序列导出目录（与 `--serve` 二选一） |
| `--serve` | 无 | 启动本地交互查看器的监听端口（仅绑定 `127.0.0.1`） |
| `--difix` | 无 | 可选的本地 `nvidia/difix_ref` 快照目录 |
| `--start` | `2.0` | 相对五路公共覆盖区起点的起始秒数 |
| `--frames` | `1` | 导出或查看器可访问的时间窗数量 |
| `--step` | `0.2` | 相邻输出时间窗之间的步长（秒） |
| `--yaw` | `0` | 虚拟相机相对车体的水平环绕角（度，导出模式生效） |
| `--pitch` | `12` | 虚拟相机下俯角（度，导出模式生效） |
| `--height` | `2.4` | 虚拟相机相对地面的高度参数（米，导出模式生效） |
| `--rear` | `4.5` | 虚拟相机在车后的水平距离（米，导出模式生效） |

## 6. 怎么看效果与测速

1. 先选低位追车（如车后 `4.5 m`、高度 `2.4 m`、下俯 `12°`）。查看器下方并排显示五路同一时刻的实拍缩略图，可用道路边缘、车辆和围栏作为参考基准。
2. 暂停在某一时刻，调整页面上的环绕角、俯角、高度或后移距离，确认新画面更新。保持机位不变再切换 Difix，对比线条清晰度与车辆轮廓变化。
3. 抬高机位，观察新露出的墙后、围栏后方和屋顶区域。缺少目标机位实拍时，纹理变自然只代表观感变化，不能据此认定遮挡区几何或内容正确。
4. 测速时建议分别记录：**固定时刻换机位**、**连续推进新时刻出图**、**跳转时间窗后首帧**。查看器对所有机位现场渲染，状态栏显示从前端发出请求到图像解码完成的总耗时。

## 7. 依赖边界与许可说明

- **上游接口依赖**：[render_view.py](render_view.py) 直接调用了上游 `NCoreInstantNuRecDataset` 的部分内部数据加载方法（`_get_loaders_and_sensors`、`_build_camera_subsampler`、`_get_rig_trajectory`、`_load_data_batch`）以及 `model.reconstruct`。若上游代码升级导致接口签名变化，请检出已验证的上游 commit `f2ecafcdf28f2dc8f2cc17447a772571cde558be`。
- **上游代码、依赖、模型与数据许可**：本仓库仅包含应用层推理与查看器胶水代码，**不分发任何第三方源码、模型权重或数据集**。使用与分发前请分别遵循各上游组件条款：

| 组件 | 来源 | 许可协议 | 核心约束说明 |
| --- | --- | --- | --- |
| `NVIDIA/instant-nurec` 代码 | [GitHub 仓库](https://github.com/NVIDIA/instant-nurec/blob/f2ecafcdf28f2dc8f2cc17447a772571cde558be/LICENSE.txt) | Apache License 2.0 | `Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES`；详见上游 [LICENSE.txt](https://github.com/NVIDIA/instant-nurec/blob/f2ecafcdf28f2dc8f2cc17447a772571cde558be/LICENSE.txt) |
| `instant-nurec` 渲染扩展依赖（含 `nvdiffrast` 等） | [THIRD_PARTY_LICENSE.txt](https://github.com/NVIDIA/instant-nurec/blob/f2ecafcdf28f2dc8f2cc17447a772571cde558be/THIRD_PARTY_LICENSE.txt) | 含 NVIDIA Source Code License (1-Way Commercial) 等 | 执行 `uv sync --extra render` 会安装 `nvdiffrast`；非 NVIDIA 使用仅限非商业研究或评估，安装或再分发前请先阅读完整第三方声明 |
| `nvidia/instant-nurec`（`pa-multiview` 权重） | [Hugging Face 模型卡](https://huggingface.co/nvidia/instant-nurec) | NVIDIA Open Model License | 遵循模型卡页面列明的模型使用条款 |
| `nv-tlabs/Difix3D` 代码与 `nvidia/difix_ref` 权重 | [Difix3D 仓库](https://github.com/nv-tlabs/Difix3D) · [Hugging Face 模型卡](https://huggingface.co/nvidia/difix_ref) | NVIDIA License | 仅限研究、开发与非商业用途；涉及商业使用请勿直接采用 |
| `PhysicalAI-Autonomous-Vehicles-NCore` 示例数据 | [Hugging Face 数据集](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore) | NVIDIA 数据集条款 | 需个人账号登录并在页面接受条款后下载 |
