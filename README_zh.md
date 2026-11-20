# LingSante-Pulse

[English](README.md) | [简体中文](README_zh.md)

面向生理时序的表征学习与建模工具，提供生理波形编码和连续血糖（CGM）分析能力。

LingSante-Pulse 帮助研究者和算法开发者从生理信号中提取可复用的表征。项目既提供 ECG、EEG、PPG 等生理波形的预训练编码能力，也通过 CGM 分支支持连续血糖的日内特征提取、跨日历史建模、公开数据预训练和下游任务评测。

[技术报告](docs/LingSante-Pulse_Technical_Report.pdf)

## 主要功能

| 功能分支 | 输入 | 输出 | 支持的工作流程 |
| --- | --- | --- | --- |
| 生理波形 | 信号窗口、实际采样率、通道信息 | 每通道 256 维 CLS / PatchMean，可选 patch 序列 | 预训练表征提取，供下游任务使用 |
| 连续血糖（CGM） | 按日整理的血糖、观测掩码、受试者与时间信息 | 每日 146 维表征及所用历史窗口数 | 表征提取、公开数据预训练、下游分类、固定模型与候选模型选择评测 |

CGM 分支当前采用 Transferable Evidence Reconstruction（TER）方法，结合日内表征、物理尺度、动态变化与因果跨日历史。该分支在项目统一的安装、文档和版本管理下，保留完整的训练与评测流程。

各分支提供符合数据特性的预处理：波形分支处理信号采样率和物理通道，CGM 分支处理血糖单位、日内采样网格及受试者历史。对应入口提供默认参数和预训练权重加载，用户可直接从数据开始使用。

## 安装

建议使用 Python 3.11 或 3.12。以下命令均从仓库根目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install . --no-deps
```

根目录的 `requirements.txt` 汇总项目各功能分支所需依赖；CGM 评测的版本约束沿用其已有配置。GPU 训练需使用与设备匹配的 PyTorch 构建。CPU 可以运行波形和 CGM 分支的推理及合成数据示例。

## 预训练权重

| 功能分支 | 获取方式 | 内容 |
| --- | --- | --- |
| 生理波形 | [Hugging Face：inclusionAI/LingSante-Pulse-1.0](https://huggingface.co/inclusionAI/LingSante-Pulse-1.0) | 正式编码器、配置和固定通道资产 |
| CGM | 仓库中的 `src/lingsante_pulse/cgm/assets/checkpoints/` | 日编码器、动态分支、多日模型和物理尺度参数 |

波形编码器默认从 Hugging Face 下载并使用本地缓存，也支持加载本地模型目录。CGM 部署权重随仓库和安装包提供。

## 快速开始

### 生理波形：提取信号特征

加载波形编码器并提取特征：

```python
import numpy as np
from lingsante_pulse import LingSantePulse

model = LingSantePulse.from_pretrained()

# 两个合成 ECG 窗口，单通道，每个窗口 10 秒，采样率 128 Hz。
signals = np.random.default_rng(42).normal(size=(2, 1, 1280)).astype(np.float32)
features = model.encode(
    signals,
    sample_rate=128,
    channels=[{"modality": "ECG", "lead": "II"}],
)

print(features["cls"].shape)         # torch.Size([2, 256])
print(features["patch_mean"].shape)  # torch.Size([2, 256])
```

使用本地模型目录时，将加载语句替换为：

```python
model = LingSantePulse.from_pretrained("models/pulse", local_files_only=True)
```

自定义本地目录需要包含 `model.safetensors`、`config.json`、`channels.npz` 和 `channels.json`。也可向 `from_pretrained` 显式传入 Hugging Face 仓库 ID。

### CGM：提取连续血糖特征

CGM 分支权重与合成数据示例随仓库提供，可直接运行：

```bash
python -m lingsante_pulse.cgm extract \
  --input examples/cgm_synthetic.npz \
  --output results/cgm_example_embeddings.npz
```

CGM 入口默认加载随安装包提供的部署权重，在 CPU 上推理。输出包含 `embedding`、`history_days` 及对应的窗口、受试者和时间信息。输出文件已存在时，命令会报错，重新运行请指定新的输出路径。

Python 调用方式：

```python
import numpy as np
from lingsante_pulse.cgm import extract

with np.load("examples/cgm_synthetic.npz", allow_pickle=False) as archive:
    payload = dict(archive)

embedding, history_days = extract(
    payload,
    device="cpu",
)

print(embedding.shape)     # (6, 146)
print(history_days.shape)  # (6,)
```

## 输入与输出

### 生理波形分支

- 数组支持 `[T]`、`[C, T]`、`[B, C, T]`；单通道批量输入使用 `[B, 1, T]`。也支持按通道组织的字典输入。
- `sample_rate` 应填写实际采样率。波形使用固定的 128 Hz 预处理；输入通道需在时间上对齐。
- `channels` 按输入顺序填写模态与可选导联。例如 `{"modality": "ECG", "lead": "II"}`。
- 名称忽略大小写和首尾空格。ECG 的 `II`、`ii`、`"2"` 和 `2` 均映射为 II 导联；省略或留空导联时使用通用 ECG 身份。
- 仅接受模型支持的通道身份。可通过 `model.channel_names` 和 `model.channel_descriptions` 查询；未支持的身份（如 IMU）会报错。
- 可提供与输入形状一致的 `observed` 和 `sample_present` 布尔掩码，分别表示有效观测和存储填充。

| 输出 | 形状 | 含义 |
| --- | --- | --- |
| `cls` | `[B, C*256]` | 各通道 CLS 特征，按输入顺序拼接 |
| `patch_mean` | `[B, C*256]` | 各通道有效 patch 的平均特征 |
| `channel_names` | C 个名称 | 规范化后的通道身份 |
| `patch_tokens` | `[B, P, C*256]` | 设置 `return_tokens=True` 时返回的 patch 特征 |
| `patch_valid` | `[B, P, C]` | 可选的逐通道 patch 有效性掩码 |
| `patch_times` | `[P]` | 可选的 patch 相对起始时间，单位为秒 |

多通道在编码器中独立处理，随后拼接特征。波形采用逐通道、逐窗口的有效观测均值与标准差归一化。数值通道使用零阶保持重采样，并按下表的固定物理尺度转换：

| 数值通道 | 输入单位 | 中心值 | 缩放值 |
| --- | --- | --- | --- |
| `spo2` | 百分数，例如 97 | 95 | 5 |
| `heart_rate`、`pulse_rate` | 次/分钟 | 70 | 30 |
| `rr_primus`、`rr_solar` | 次/分钟 | 20 | 20 |
| `bis_emg_power` | dB | 0 | 100 |

数值按 `(value - 中心值) / 缩放值` 转换并截断至 `[-10, 10]`。NaN/Inf 视为未观测，零值本身是合法采样；完全没有有效观测的窗口会报错。

### CGM 分支

CGM 分支输入为非重叠的 24 小时血糖窗口，每个窗口使用 5 分钟网格，共 288 个点：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `glucose_mg_dl` | `[N, 288]` | 血糖值，单位为 mg/dL |
| `observed_mask` | `[N, 288]` | 原始物理观测的布尔掩码，插值点不标记为观测 |
| `window_id` | `[N]` | 唯一窗口标识 |
| `subject_id` | `[N]` | 匿名受试者标识，用于划分历史，不作为数值特征 |
| `start_time` | `[N]` | 真实窗口起始时间，保留当地日内时钟约定 |
| `episode_index` | 可选 `[N]` | 同一受试者的不同记录片段，避免跨片段拼接历史 |

将同一受试者、同一片段的可用历史窗口放在同次调用中。模型按时间组织最多七天的因果上下文；没有可用历史时返回当前日表征。若存在原始采样间隔信息，保留数据构建器提供的 `native_cadence_min` 等字段。

输入若使用 mmol/L，需在构建窗口前转换为 mg/dL。可附带 `label_*` 字段供独立评测使用，编码器不消费这些标签。完整字段和数据构建规则见 [CGM 数据说明](docs/cgm/DATA.md)。

## CGM 分支：下游分类与预测

先分别提取训练受试者和新受试者的特征，并保留各自可用的历史。使用训练受试者的标签拟合分类器：

```bash
python -m lingsante_pulse.cgm downstream fit \
  --input results/train_embeddings.npz \
  --label label_diabetes \
  --model results/diabetes_classifier.joblib

python -m lingsante_pulse.cgm downstream predict \
  --input results/new_participants_embeddings.npz \
  --model results/diabetes_classifier.joblib \
  --output results/diabetes_predictions.npz
```

这些命令需要用户自己的特征文件和训练标签。分类示例使用 `StandardScaler + LogisticRegression`，归一化参数与分类器均仅在训练受试者上拟合；加载自己生成或可信来源的分类器文件。预训练权重本身输出表征，不包含已训练好的疾病分类器。

## CGM 分支：公开数据预训练

仓库保留数据准备和多阶段训练代码。按 [CGM 数据说明](docs/cgm/DATA.md) 获取公开数据，并使用随附构建器生成训练语料。

```bash
python -m lingsante_pulse.cgm prepare-data \
  --shanghai-root data/raw/shanghai \
  --stanford-root data/raw/stanford \
  --big-ideas-root data/raw/big_ideas_1.1.2 \
  --colas-root data/raw/colas2019/S1 \
  --output-dir data/processed/public_pretraining \
  --shanghai-policy subject_disjoint --seed 43

python -m lingsante_pulse.cgm train \
  --corpus data/processed/public_pretraining/glucofm_public_pretrain_subject_disjoint_v1.npz \
  --output results/cgm_public_pretraining \
  --device cuda:0
```

将 `data/raw/` 下的示例路径替换为实际取得的数据目录。训练依次执行日编码器、动态分支、跨日上下文和最终记忆阶段，结束后在 `results/cgm_public_pretraining/checkpoints/` 导出可推理权重。当前公开配方固定 seed 43，每个优化阶段运行 6,960 步，需要 CUDA 兼容加速器。

训练语料不接收表型标签，训练和验证受试者互斥。随库提供的正式 CGM 分支权重还使用过私有无标签 CGM；上述命令复现公开数据训练流程，不会重建包含私有数据的原始权重。训练细节与模型模块说明见 [CGM 代码说明](docs/cgm/CODE_MAP.md)。

## CGM 分支：评测

公开评测覆盖 CGMacros、Shanghai、Stanford 和 Hall 四个队列，共 14 个任务。先按 [CGM 数据说明](docs/cgm/DATA.md) 构建各队列数据，组织为 `data/processed/glucofm_public/COHORT/COHORT_glucofm_windows_v1.npz`。其中 `COHORT` 分别为 `cgmacros`、`shanghai_t2dm`、`stanford` 和 `hall`。

### 固定模型评测

评测随库部署权重：

```bash
python -m lingsante_pulse.cgm evaluate \
  --data-root data/processed/glucofm_public \
  --output results/cgm_fixed_model \
  --device cpu
```

评测自己训练的模型时，增加 `--checkpoint-dir` 参数，指向新训练输出的 `checkpoints/` 目录。也可以通过 `--embeddings-root` 评测已经提取好的各队列特征。

### 可选的论文主表复现

论文主表采用 24 个候选模型的嵌套选择流程，需额外提供候选权重及 `manifest.json`。默认安装包提供单个部署模型。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m lingsante_pulse.cgm evaluate-nested \
  --data-root data/processed/glucofm_public \
  --candidates-dir reproduction/candidates \
  --output results/cgm_nested_evaluation \
  --device cpu --workers 8
```

将 `--candidates-dir` 指向准备好的候选目录。流程保持 10 次重复、5 折外层与 3 折内层评测：在每个外层训练集内部选择一套共享表征，再评估外层测试集。同一生物受试者的记录始终归入同一折。固定部署模型的评测成绩与该候选选择流程的主表成绩分别报告。

## 仓库结构

```text
.
├── README.md
├── README_zh.md
├── LICENSE
├── LEGAL.md
├── pyproject.toml
├── requirements.txt
├── src/lingsante_pulse/
│   ├── __init__.py
│   ├── waveform/
│   │   ├── api.py
│   │   ├── model.py
│   │   ├── preprocessing.py
│   │   ├── channels.py
│   │   └── hub.py
│   └── cgm/
│       ├── __main__.py
│       ├── models/
│       ├── data/
│       ├── inference/
│       ├── training/
│       ├── evaluation/
│       ├── downstream.py
│       ├── checkpoints.py
│       ├── baselines/
│       ├── diagnostics/
│       └── assets/
│           ├── checkpoints/
│           ├── configs/
│           └── benchmarks/
├── examples/
├── scripts/cgm/
├── tests/
│   ├── waveform/
│   └── cgm/
└── docs/cgm/
```

## 许可证与数据来源

项目自有代码与随附模型权重采用 MIT 许可证，保留仓库的 `LEGAL.md` 及 CGM 分支的第三方来源说明。第三方数据和模型按照各自的许可与访问条件获取，不随本仓库重新分发。私有原始数据、表型标签和个人身份信息不包含在发布内容中。

LingSante-Pulse 用于表征学习与算法研究，具体任务的预测效果需要在目标数据及相应评测协议下验证。
