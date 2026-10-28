# lamet-agent

<p align="right">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <img src="docs/lamet-agent-demo.gif" alt="lamet-agent 演示" width="800" />
</p>

`lamet-agent` 是一个 Python-first 框架，用于构建可复现的**大动量有效理论**（**La**rge **M**omentum **E**ffective **T**heory，LaMET）和格点 QCD 分析工作流。

## 快速开始

本机需要已登录的 Codex CLI。Codex 不使用
`--api-key-file`。

```bash
git clone https://github.com/AI4LGT/lamet-agent.git && cd lamet-agent
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip && python3 -m pip install -e ".[codex]"
```

请使用百度网盘下载原始数据，链接为：https://pan.baidu.com/s/1I31_YJMbrkmShYgDRsXGtw?pwd=AI4L
。请自行解压缩 data_lamet_agent.zip 包，进入该目录后解压缩 data_pion_pdf_cg.zip 包。
最后将文件夹 data_pion_pdf_cg 和文件 plot_pion_pdf_compare.py 放入 lamet-agent 根目录下。
```bash
lamet-agent run examples/pion_pdf_cg_manifest.json \
  --provider codex --model gpt-5.6-luna
```

如需将本次运行结果与参考结果进行比较：

```bash
python plot_pion_pdf_compare.py
```

也可以使用 `uv` 创建环境并安装：

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[codex]"
```

### 其他提供商

API 提供商（`openai`、`anthropic`、`gemini`、`grok`、`deepseek`，或自定义的兼容 OpenAI 的 HTTP(S) URL）需要执行 `python3 -m pip install -e .`（不使用 `[codex]` extra），并通过 `--api-key-file` 或相应提供商的环境变量提供密钥。详情参见[提供商和模型](#提供商和模型)。

```bash
lamet-agent run examples/pion_pdf_cg_manifest.json \
  --provider openai --model gpt-5.6-luna \
  --api-key-file api.key
```

### Manifest 示例

| Manifest                                     | 工作流                                                         | 数据参考                                                                                     |
| -------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `examples/pion_pdf_cg_manifest.json`         | Coulomb-gauge pion PDF，使用最小二乘关联函数分析。              | [data_pion_pdf_cg.zip](https://pan.baidu.com/s/1I31_YJMbrkmShYgDRsXGtw?pwd=AI4L)[^1]         |
| `examples/pion_pdf_cg_lanczos_manifest.json` | Coulomb-gauge pion PDF，使用 nested-bootstrap Lanczos 分析。    | [data_pion_pdf_cg.zip](https://pan.baidu.com/s/1I31_YJMbrkmShYgDRsXGtw?pwd=AI4L)[^1]         |
| `examples/pion_pdf_gi_manifest.json`         | Gauge-invariant pion PDF。                                     | [data_pion_pdf_gi.zip](https://pan.baidu.com/s/1I31_YJMbrkmShYgDRsXGtw?pwd=AI4L)[^1]         |
| `examples/pion_da_gi_manifest.json`          | Gauge-invariant pion DA，包含系统学变体。                       | [data_pion_da_gi.zip](https://pan.baidu.com/s/1I31_YJMbrkmShYgDRsXGtw?pwd=AI4L)[^2]          |
| `examples/kaon_da_gi_manifest.json`          | Gauge-invariant kaon DA，包含系统学变体。                       | [data_kaon_da_gi.zip](https://pan.baidu.com/s/1I31_YJMbrkmShYgDRsXGtw?pwd=AI4L)[^2]          |

这些 zip 都在[百度网盘](https://pan.baidu.com/s/1I31_YJMbrkmShYgDRsXGtw?pwd=AI4L)的 `data_lamet_agent.zip` 里。请选择与该 manifest 所使用的 `data_*` 目录相匹配的 zip 文件（`data_pion_pdf_cg`、`data_pion_pdf_gi`、`data_pion_da_gi`、`data_kaon_da_gi`）。

[^1]: Xiang Gao, Wei-Yang Liu, and Yong Zhao, [*Parton Distributions from Boosted Fields in the Coulomb Gauge*](https://arxiv.org/pdf/2306.14960), arXiv:2306.14960.
[^2]: Jun Hua et al., [*Pion and Kaon Distribution Amplitudes from Lattice QCD*](https://arxiv.org/pdf/2201.09173), arXiv:2201.09173.

## 命令行

### C32P23 bootstrap TMDWF 最小接入示例

#### 使用现有拟合、λ 延拓和傅里叶接口

`examples/c32p23_native_p4_p1_2_b1.json` 接到正常 agent job loop，依次运行：

1. `correlator_analysis` 的 `qda_ratio`：四个 `gtg5/gzg5 × zdir0/zdir1` 通道各自拟合。
2. `renormalization`：使用 `strategy=wilson_loop`，先 `type=fit` 拟合中心因子，再 `type=apply` 平均四个已拟合的通道并重整化；`normalization=false`。
3. `fourier_transform`：通过 `tmdwf_projection=real/imag` 在本 stage 内准备相位投影，然后使用现有 LA/NLA 尾部拟合、λ 延拓、平滑和傅里叶扫描。
4. 启动脚本调用 `fourier_transform.physics.combine_tmdwf_transforms`，按 bootstrap 编号组合成复数 `quasi_tmdwf.nc`，不另做拟合或变换。

不再注册独立 TMDWF stage。通道平均函数属于 `correlator_analysis.physics`；Wilson 拟合、分母应用属于
`renormalization.physics`；相位投影和复数结果合并属于 `fourier_transform.physics`。JSON 只声明上述三个 stage。

已准备的输入在 `data_C32P23/native/p4_p1_2_b1/`，其中 `correlators.h5` 同时含有两点函数
和 Wilson-loop 数据。主文件 `data_C32P23/correlators/C32P23/2pt/C32P23_2pt_bootstrap.h5`
也包含完整 Wilson-loop 组。两份 HDF5 都使用以下附加结构，原有两点函数路径保持不变：

```text
wilson_loops/C32P23/
  raw                 (451, 3, 26, 11): configuration, direction, length_lattice, b_perp_lattice
  configuration       451 个实际组态编号
  direction           0..2
  length_lattice      0..25
  b_perp_lattice      0..10
  mean                (26, 11): 对组态及方向平均的未拟合 Wilson loop
  source_text         原始 txt 字节；组属性保存 SHA-256
```

Wilson 组明确标记 `sample_kind=raw_configuration`；不把它的 451 个组态当成两点函数的
1000 个 bootstrap 样本。Wilson 的原始组态用 leave-one-out jackknife 估计均值的对角误差，
对均值拟合一次；拟合后仅使用固定中心值作为全部 bootstrap 样本共同的分母，
这是本分析明确选择的处理方式，不传播 Wilson-loop 或 `ZO` 误差。
新增导入命令如下；重复导入相同内容安全，冲突内容会拒绝覆盖：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B scripts\add_c32p23_wilson_h5.py data_C32P23\wilson_loops\wiloop_hyp1_C32P23.txt data_C32P23\correlators\C32P23\2pt\C32P23_2pt_bootstrap.h5
```

若在新目录准备 native 输入，先完成 Wilson 导入，再运行
`python -X utf8 -B scripts/prepare_c32p23_native.py`；它会连同 Wilson 组一起复制，
遇到已有目标文件会停止，避免覆盖。
非局域分子保留复数，分母为同一 source／方向的 `b=0,z=0` 局域实部。
descriptor 的 `sampling={"mode":"bootstrap","resample_id":"..."}` 显式标记已有样本，
`dims` 首维为 `bootstrap`；`count` 和 `metadata.samples` 均为 1000，`bin_size=1`。
系统校验样本 ID、数量和模式，不重新 bootstrap。未写 `sampling` 的原有 descriptor 仍走 raw 抽样流程。

从项目根目录进行纯本地数值接入测试（建议由固定测试参数提供，不调用 LLM）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B scripts\run_c32p23_native.py examples\c32p23_native_p4_p1_2_b1.json --fixed-suggestions
.\.venv\Scripts\python.exe -X utf8 -B scripts\verify_c32p23_native.py runs\c32p23_native_p4_p1_2_b1\artifacts_offline
```

真实模型测试配置为 `examples/c32p23_native_p4_p1_2_b1_agent.json`，时间窗口、`zmin_fm` 和
`zmax_fm` 均设为 `null`，由原有推荐接口向模型请求。允许 provider 接收均值、误差及拟合诊断后运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B scripts\run_c32p23_native.py examples\c32p23_native_p4_p1_2_b1_agent.json --provider codex --model gpt-6.1-sol
```

也可用正常 `lamet-agent run` 执行该 manifest，再用启动脚本的 `--combine-only` 组合两路输出。
离线测试写入 `runs/c32p23_native_p4_p1_2_b1/artifacts_offline`；上述真实 provider 配置写入
`runs/c32p23_native_p4_p1_2_b1_agent/artifacts`，两者分开保存。
迁移前的真实 Codex 测试完成了 9 个 job 和 6 次模型建议调用；旧输出保留作为参考。
当前合并版有 8 个 job（两路 Fourier 共用一次重整化），范围为 null 时仍使用原有模型推荐接口。
本次迁移的完整离线回归输出位于 `runs/c32p23_existing_stages_verified/artifacts_offline`，数值检查见其中的 `native_verification.json`。
重复运行前请在 manifest 中设置新的 `artifacts_directory`，以保留原结果。

这里的 `tmdwf_profile=pion_da_components` 是明确选择的尾部模型适配：延续旧 notebook 的
midpoint 实部、虚部均为偶函数的负 z 约定；分别取实部和虚部振幅、乘 `exp(-i*λ/2)`，
再使用原有 light-light DA 尾部和 `exp(+i*x*λ)` 核。两路最后按 `F_re + i*F_im` 合成，
不会丢弃 TMDWF 虚部。它复用现有算法，但不表示 TMDWF 等同于 DA，也不调用 DA matching。
与旧 notebook 的先平均再拟合、六参数联合尾部模型不同，数值无需逐点相等。

验证范围为 `Pz=4,p1=2,p2=-2,b=1`，物理时间坐标为 `t=1..15`，拟合窗口右端不包含。
合并版的完整本地回归运行 8 个 job，保留 1000 个样本和 1000 个 x 点。
全量数据入口和复数 Fourier 相位恒等式检查见 `native_verification.json`。
迁移前后同参数的四路拟合、Wilson 因子、两路延拓和最终复数数组逐元素相同，检查见 `stage_migration_regression.json`。
Wilson-loop 和 `ZO` 按本分析约定使用固定中心值；结果尚未进行硬核匹配。
本例验证接口可运行，后续需要比较时间窗口、态数和尾部模型的系统误差。

#### 已迁移的旧接口

旧的独立 TMDWF stage、notebook 专用拟合／傅里叶实现、启动脚本及 manifest 已退出当前源码。
本地迁移备份在 `runs/tmdwf_stage_migration_backup/`，旧计算结果保留；运行当前版本请使用上面的 native manifest。
这次重构迁移必要功能，保留现有 ratio 拟合及 LA/NLA 扫描算法，不再维护另一套 notebook 数值流程。

`lamet-agent` 和 `python -m lamet_agent` 提供相同的接口：

```text
lamet-agent {validate,plan,run} ...
```

### 验证

```bash
lamet-agent validate MANIFEST
```

Validate 读取 JSON 或 JSONC，并且不与 LLM 通信。它检查 manifest envelope、stage contract、systematics 声明、job DAG、路径、关联函数 descriptor 和 kernel 参数。

### 规划

```bash
lamet-agent plan MANIFEST \
  --provider PROVIDER \
  [--model MODEL] \
  [--api-key-file FILE] \
  [--output FILE | --in-place]
```

Plan 通过与 LLM 进行交互式对话来补全不完整的 manifest。它会验证提议的修改，并给出最终的自然语言摘要，供用户明确确认。

选项：

- `--provider`：已注册的提供商名称，或兼容 OpenAI 的 HTTP(S) URL；
- `--model`：模型 ID；省略或不可用时会提示选择；
- `--api-key-file`：只包含 API key 的文本文件；
- `--output`：输出路径；
- `--in-place`：在用户明确接受后覆盖源文件。

`--output` 与 `--in-place` 互斥。规划后的输出必须与源 manifest 位于同一目录，以保持相对输入路径的含义不变。
如果两者都未设置，Plan 会先询问输出文件名，并相对于源 manifest 所在目录解析。终端将该目录显示为相对于当前工作目录的固定前缀。只有文件名可以编辑，并预填为 `<manifest>.planned.json`；
可以修改，或按 Enter 接受。清空输入即原地保存；在纯文本模式下，空输入同样选择原地保存。任何指向源文件的输出都需要单独确认是否覆盖（默认为否），包括 `--in-place`。
保存仍需要最终确认。

终端控制命令包括 `/show`、`/issues`、`/undo`、`/edit`、`/save`、`/help` 和 `/quit`。`Enter` 提交，`Shift+Enter` 插入换行，`Ctrl+C` 取消。

独立运行 Plan 时，会写入已接受的 manifest，然后退出，不会执行分析 stages。
Plan 默认不保存 LLM transcripts。向 `plan` 或 `run` 传入 `--plan-log-dir DIRECTORY` 即可启用。该目录相对于 CLI 工作目录解析；每次 Plan 会话都会得到一个以 manifest 名称和时间戳命名的子目录。
界面会打印 transcript 的相对路径。它记录 requests、responses、被拒绝的回复、失败原因和耗时，并且不会覆盖先前的会话。

### 运行

```bash
lamet-agent run MANIFEST \
  --provider PROVIDER \
  [--model MODEL] \
  [--api-key-file FILE] \
  [--progress {auto,stage,job,none}]
```

Run 在执行数值 stages 前先进行验证。如果验证失败，它会使用所选提供商进入 Plan。用户接受修复后的有效 manifest 后，数值执行将自动继续。

默认进度模式为 `auto`：

- `auto`：声明了 systematics 时显示 stage-level job progress；否则进度由各数值 job 自行管理；
- `stage`：每个 stage 显示一个 job 计数器；
- `job`：显示特定 stage 的数值进度；
- `none`：禁用进度条。

### 提供商和模型

#### Codex CLI

`codex` 提供商使用可选的 `openai-codex>=0.147` 包以及当前机器上缓存的 Codex 登录状态。它不使用 API key。`--model` 是可选项；可用模型从 Codex app server 发现。

#### 兼容 OpenAI 的 API

已注册的 API 提供商包括 `openai`、`anthropic`、`gemini`、`grok` 和 `deepseek`。每个提供商都从 `--api-key-file` 或相应环境变量中读取 API key：

| 提供商    | 环境变量            |
| --------- | ------------------- |
| OpenAI    | `OPENAI_API_KEY`    |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini    | `GEMINI_API_KEY`    |
| Grok      | `GROK_API_KEY`      |
| DeepSeek  | `DEEPSEEK_API_KEY`  |

所有提供商都会在选择前发现可用模型：API 提供商使用 `/models`，Codex 使用 app-server 的 `model/list`，Claude Code 使用 SDK server 的初始化清单（包括别名和已解析的 model ID）。省略 `--model`，或提供一个不可用的 ID，即可在共用界面中按编号或名称选择模型。即使只有一个可用模型，也不会自动选定。库的调用方可以向 `create_backend` 传入 `select_model` 回调；如果没有该回调，缺失或不可用的模型会报错并列出可用选项。也可以直接把兼容 OpenAI 的 HTTP(S) base URL 作为提供商传入。

## 核心思想

manifest 包含运行元数据以及按顺序排列的 stage job 列表映射。Job id 构成一个 DAG：关联函数 jobs 选择原始记录，后续 jobs 通过 `target`、`denominator`、`input` 和 `quasi` 等按角色命名的 inputs 使用先前输出。

预期的 agent 行为：

- 在执行数值计算前验证完整的已编写工作流。
- 参数确定后，以确定性方式运行数值 stages。
- 仅在工作流需要 fit 或区间建议时咨询 LLM。
- 写出中间 NetCDF 数据、诊断信息、图和 stage reports，使完整分析路径可以检查。
- 基于数值证据、一致性检查和选定文献完成最终 Review。

已实现的 stage families 通常按以下顺序编写：

1. `correlator_analysis`
2. `renormalization`
3. `fourier_transform`
4. `perturbative_matching`
5. `extrapolation`
6. `review`

部分工作流可以省略不需要的 stages。`stages` 下各键的顺序就是执行顺序；不存在单独的 `metadata.stages` 列表。

架构、文件所有权和贡献者工作流记录在 [`DEVELOPMENT.md`](DEVELOPMENT.md) 中。

## 中间数据（NetCDF）

stage 之间传递的数值 artifacts 存储为 NetCDF 文件。每个 array 都具有：

- 最前面的 `resample` 维度；
- 一种采样模式：`raw`、`jackknife`、`bootstrap` 或 `gvar`；
- `t`、`tsep`、`tau`、`z`、`x`、`a` 或 momentum 等物理维度和坐标；
- 以 attributes 形式存储的 ensemble 和 stage provenance。

Job 也可以使用外部 `{ "file": ".../output.nc" }` artifact 作为输入。

典型的每个 job 文件包括：

| 文件                  | 用途                                             |
| --------------------- | ------------------------------------------------ |
| `output.nc`           | 携带 samples 的主要数值结果。                    |
| `summary.json`        | 决策、诊断信息和已声明的 artifacts。             |
| `llm_transcript.md`   | 适用时记录 LLM requests 和 responses。           |
| `diagnostics/*`       | 候选表和数值诊断。                               |
| `plots/*`             | PDF/SVG 结果和 fit-quality 图。                  |

stage 目录还会获得汇总的 `report.md`；job 目录不写 report 文件。Review 会写出最终的 `review.md`、`review_bundle.json` 以及一致性/文献证据。

### 不依赖 lamet-agent 检查或读取数据

NetCDF 是自描述格式，可以使用 `ncdump`、Panoply 或 xarray 检查：

```python
import xarray as xr

array = xr.load_dataarray("output.nc", auto_complex=True)
print(array.dims)
print(array.coords)
print(array.attrs)
```

第一维始终是 `resample`；其余维度描述相应 stage report 中记录的物理布局。

## Manifest 示例

loader 接受 JSON 和 JSONC 注释。当前 manifest envelope 为：

```json
{
  "metadata": {
    "run_id": "pion_pdf_cg",
    "root_directory": "..",
    "artifacts_directory": "runs/pion_pdf_cg/artifacts",
    "random_seed": 1984,
    "workers": 4,
    "target_observable": "pdf",
    "parton": "quark",
    "resample_mode": "jackknife",
    "bin_size": 1,
    "sample_error_mode": "covariance"
  },
  "stages": {
    "correlator_analysis": {
      "defaults": {},
      "jobs": [
        {
          "id": "ca_p5",
          "inputs": {
            "correlators": [
              {"json": "examples/pion_pdf_cg_correlators.json", "id": "p5_2pt"},
              {"json": "examples/pion_pdf_cg_correlators.json", "id": "p5_3pt"}
            ]
          }
        }
      ]
    }
  },
  "systematics": {}
}
```

三个顶层 objects 分别是：

- `metadata`：全局运行路径、target identity、重采样、误差、seed 和 worker 数量；
- `stages`：有序的 stage/job graph；
- `systematics`：可选的、由 stage 所有的变体声明。

每个 stage 包含共享的 `defaults` 和有序的 `jobs` 列表。一个 job 包含全局 `id`、按角色命名的 `inputs`，以及直接写在 job 上的参数覆盖。Stage defaults 填充 job 中省略的字段；显式 job 值仍具有最高优先级。

输入值可以是：

- 先前 job 的 id；
- `{ "file": "path/to/output.nc" }`；
- `{ "json": "descriptor.json", "id": "correlator_record" }`；
- receiving contract 允许的数值常量；
- receiving role 允许多个来源时使用的有序列表。

对于使用 descriptor JSON record 的关联函数输入，请参阅[标准关联函数 HDF5 格式](#标准关联函数-hdf5-格式)中的输入文件约定。

未知字段、无效选项、损坏的 input roles、重复 id、向前 job 引用、缺失路径以及跨参数不一致都会被拒绝。

全局运行 metadata 字段包括：

- 必填：`run_id`、`root_directory`、`artifacts_directory`、`random_seed`、`workers`、`target_observable`、`resample_mode`、`sample_error_mode` 和 `bin_size`；
- `parton`，当前为 `quark`，默认值也是该值；
- `samples`，仅在 bootstrap 模式下必填；
- `parameter_recommendation_retries`，默认为每个 job 额外尝试一次。

`target_observable` 接受 `pdf`、`da`、`gpd` 及上述局部扩展的 `tmdwf`。`sample_error_mode` 接受 `covariance`、`variance` 以及仅用于 bootstrap 的 `one_sigma`。

目前 Fourier、matching 和 extrapolation 支持系统学变体。这些变体会在执行前展开为具体 jobs，并保存到 `resolved_manifest.json`。

## 标准关联函数 HDF5 格式

关联函数 jobs 从 descriptor JSON 中选择一条记录：

```json
{"json": "examples/pion_pdf_cg_correlators.json", "id": "p5_3pt"}
```

### Descriptor 示例

下面这个完整示例定义了一个两点关联函数：

```json
{
  "correlators": [
    {
      "id": "p5_2pt",
      "ensemble": {"series": "HISQ", "id": "HISQa060_X", "a_s": 0.06,
                   "a_t": 0.06, "L_s": 48, "L_t": 64, "m_pi": 0.3},
      "count": 109,
      "format": "hdf5",
      "path": "correlators/pion_2pt.h5",
      "dataset": "g5/g5/PX5PY0PZ0",
      "dataset_dims": ["t", "configuration"],
      "dims": ["configuration", "t"],
      "coords": {"t": [0, 1, 2, 3]},
      "selectors": {"source_operator": "g5", "sink_operator": "g5",
                    "momentum": "PX5PY0PZ0", "gfix": "CG"},
      "correlator_type": "two_point",
      "hadron": {"name": "pion"},
      "source_momentum": [5, 0, 0],
      "sink_momentum": [5, 0, 0],
      "current": null,
      "source_sink_separation": null
    }
  ]
}
```

`path` 给出相对于 descriptor JSON 的 HDF5 文件位置；`dataset` 给出其中 array 的位置：

```text
pion_2pt.h5
└── g5
    └── g5
        └── PX5PY0PZ0     dataset, shape (4, 109)
```

叶节点 axes 为 `(t, configuration)`，与 `dataset_dims` 的声明一致。

### Dataset 路径和维度

`dims` 以 `configuration` 开头，而 `coords` 提供其他所有 axes。`dataset_dims` 给出每个 HDF5 叶节点中存储的 axis 顺序。`dataset` template 可以将 coordinate values 放入 HDF5 路径：

```json
{
  "dataset": "g5/g5/gT_nonlocal/PX5PY0PZ0/tsep{tsep}/bT0/bz{z}",
  "dataset_dims": ["tau", "configuration"],
  "dims": ["configuration", "tsep", "tau", "z"],
  "coords": {"tsep": [8, 10, 12], "tau": [0, 1, 2, 3], "z": [0, 1, 2]}
}
```

它会将 `tsep` 和 `z` 展开到不同的叶节点：

```text
pion_3pt.h5
└── g5/g5/gT_nonlocal/PX5PY0PZ0
    ├── tsep8/bT0/{bz0,bz1,bz2}
    ├── tsep10/bT0/{bz0,bz1,bz2}
    └── tsep12/bT0/{bz0,bz1,bz2}
```

每个 `bz*` 叶节点存储 `(tau, configuration)`。组装后的输出顺序为 `(configuration, tsep, tau, z)`。group 名称本身不受限制；只有 `dataset` template 定义层级结构。

### 关联函数类型

| 类型          | 常见维度                      | 附加要求                                                                                  |
| ------------- | ----------------------------- | ----------------------------------------------------------------------------------------- |
| `two_point`   | `configuration, t`            | `current` 必须为 `null`。                                                                 |
| `three_point` | `configuration, tsep, tau, z` | 必须提供 `current`；应提供 `tsep` 维度或设置 `source_sink_separation`。                    |
| `qda`         | `configuration, t, z`         | 必须提供 `current`。                                                                      |

动量是整数三元组。非 null 的 `current` 恰好包含 `kernel_operator`、`parton` 和 `renormalization_scheme`。descriptor 是所有 coordinates 和 provenance fields 的权威来源。同一 job 中选定的 records 必须具有相同的 ensemble 和 configuration count。

完整 descriptors 参见 `examples/pion_pdf_cg_correlators.json` 和 `examples/pion_da_gi_correlators.json`。

## 开发

使用 uv 或 pip 安装开发依赖：

```bash
uv pip install -e ".[dev]"
# or
python -m pip install -e ".[dev]"
```

架构、文件所有权、测试和贡献者工作流记录在 [`DEVELOPMENT.md`](DEVELOPMENT.md) 中。

## 引用

如果本工作对你的研究有帮助，请引用：
```bib
@article{He:2026kft,
    author = "He, Jinchen and Jiang, Xiangyu and Yao, Fei and Zhao, Dian-Jun",
    title = "{LaMET-Agent: An Agent Framework for Large-Momentum Effective Theory Analysis}",
    eprint = "2609.32225",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    reportNumber = "FERMILAB-PUB-26-0691-T",
    month = "9",
    year = "2026"
}
```

## 相关链接

- [LQCD_Master](https://github.com/sjtu-sai-agents/LQCD_Master)（[arXiv:2607.15001](https://arxiv.org/abs/2607.15001)）
