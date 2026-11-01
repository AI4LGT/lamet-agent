# GPD Analysis Note

> 基线：当前 `main` 快照（`d5a76b6`，2026-11-01）。本文是 issue #2 要求的 GPD analysis markdown，只描述接口、公式、现状和缺口；不修改代码。
>
> 状态：**S0** 端到端完整；**S1** 通用接口可承载但缺专用实现/校验；**S2** 专用代码路径存在但缺数据/示例/集成测试；**S3** 未实现或被 contract/runtime 拒绝。
>
> 结论先行：**本快照没有 S0 的 GPD 端到端链路。** GPD 的 FT 有专用支持，matching 有一个 NLO 夸克 kernel；但 data 缺失，correlator analysis 仍是通用非前向 quark 3pt，renormalization 没有 GPD 专用因子，extrapolation 只处理固定 $(\xi,t)$ 的 $P_z\to\infty$。

## 1. 结论与范围

### 1.1 本文范围

- observable：未极化的**夸克 GPD** $H(x,\xi,t)$，以及它在有限 $P_z$ 下的 quasi-GPD；
- 强子：当前 FT/tail 代码支持 **pion** 和 **nucleon 类**（`nucleon/proton/lambda/xi0`）；**kaon GPD 未实现**；
- 不包含：gluon GPD、极化 GPD 的专用 matching、$t$ 或 $\xi$ 的外推、多 $(\xi,t)$ 联合拟合、GPD sum rule 拟合。

### 1.2 一句话状态

| stage | 状态 | 一句话原因 |
|---|---|---|
| `correlator_analysis` | **S1** | 通用非前向 quark 3pt 可拟合，但没有 GPD 专用 $\xi,t$ 语义，且无数据 |
| `renormalization` | **S1**（external）/ **S3**（GPD self） | external denominator 可做 1D $z$ 除法；无 GPD 专用 $z_{\overline{\rm MS}}$ kernel |
| `fourier_transform` | **S2** | GPD 专用 partner/phase/sector/tail 路径存在且有测试；缺数据/示例 |
| `perturbative_matching` | **S2（窄角）** | 只有 GI、$\gamma^t$、hybrid、NLO、未极化、$\xi\neq0$ 的 quark GPD kernel |
| `extrapolation` | **S1** | 通用 $h_0(x)+\sum c_t(x)B_t$；固定 $(\xi,t)$，不能外推 $t/\xi$ |
| data | **缺** | 无 GPD descriptor/示例；issue #2 由 @Greyyy-HJC 等 Qi |

> 这里的 S2 只表示“有代码路径但缺数据/端到端验证”，不表示 GPD matching 的物理完整性。

## 2. 输入、输出与跨 stage 契约

### 2.1 全局字段

```jsonc
"metadata": {
  "target_observable": "gpd",
  "parton": "quark"
}
```

- `target_observable` 是 **manifest 全局量**，不能在同一 manifest 混 GPD/PDF/DA。
- 当前 `metadata.parton` 只允许 `"quark"`；`gluon` 会在 validate 阶段被拒绝。
- GPD 的 descriptor 需要两条 2pt（source/sink 动量各一条）和一条 3pt；3pt 的 `source_momentum != sink_momentum`（非前向）。
- `gfix`（`GI`/`CG`）、`polarization`、`hadron.name` 由 descriptor 的 `selectors`/字段携带；correlator analysis 只搬运，不做 GPD 语义校验，FT 才检查。

### 2.2 数据流

```text
descriptor (2pt + 3pt)
  → correlator_analysis: h_B(z; P_i,P_f)
  → renormalization:      h_R(z; P_i,P_f)
  → fourier_transform:    F̃(x,ξ,t;P_z)
  → matching:             H(x,ξ,t;μ)
  → extrapolation:        H_0(x,ξ,t)
```

| 接口 | 写入者 | 读取者 | 关键内容 |
|---|---|---|---|
| `z` | descriptor / correlator | renorm / FT | 坐标空间分离；renorm 后单位为 fm |
| `source_momentum`,`sink_momentum` | descriptor | correlator/FT | 整数三元组；用于非前向与 $\xi,t$ |
| `current` | descriptor | correlator | 恰好 `kernel_operator,parton,renormalization_scheme` 三键 |
| `gfix`,`polarization`,`hadron` | descriptor selectors | FT | `GI/CG`、极化、强子类型 |
| `xi`,`t_gev2` | FT | matching/extrap | 代码计算并写入 attrs；CA/renorm 没有 |
| `momentum_gev` | correlator | renorm/FT | CA 是 `norm(sink_momentum)*k_s`；FT 会重写为平均 $P_z$ |
| `x` | FT | matching/extrap | 无量纲动量分数；FT 输出 `dims=["x"]` |
| `kernel_id` | matching | extrap | matching provenance；extrap 要求所有输入一致 |
| `resample` | 全链 | 全链 | leading 维；FT partner 只检查 ensemble/resample，不检查逐样本配对 |

### 2.3 descriptor 最低字段

3pt：

```jsonc
{
  "correlator_type": "three_point",
  "dims": ["configuration", "tsep", "tau", "z"],
  "coords": {"tsep": [...], "tau": [...], "z": [...]},
  "source_momentum": [0, 0, ni],
  "sink_momentum":   [0, 0, nf],
  "current": {
    "kernel_operator": "gt",
    "parton": "quark",
    "renormalization_scheme": "hybrid"
  },
  "selectors": {
    "gfix": "GI",
    "polarization": "unpolarized",
    "current_operator": "gt"
  },
  "source_sink_separation": null
}
```

2pt：`correlator_type="two_point"`、`dims=["configuration","t"]`、`current=null`，source/sink 分别匹配 3pt 的 $P_i,P_f$。

**关键限制：**

- 3pt `dims` 必须**恰好**是 `["tsep","tau","z"]`；加 `bT`/`xi`/`t` 维会被 `physics.py` 拒绝。
- 一个 job 只能有一个 3pt；两条 flow 必须拆成两个 job。
- `3pt_over_3pt`/`normalization` 要求 z 网格中**唯一**的 $z=0$。
- 当前没有 GPD data/example；`examples/` 只有 PDF/DA。

## 3. Correlator analysis

### 3.1 物理量与公式

GPD 定义（leading twist，$\Gamma=\gamma^+$）：

$$
F_\Gamma(x,\xi,t)
=\frac{1}{2}\int\frac{dz^-}{2\pi}\,
e^{ixP^+z^-}
\langle P_f|\bar\psi(-z^-/2)\Gamma\mathcal W_{z^-}\psi(z^-/2)|P_i\rangle .
$$

quasi-GPD：

$$
\widetilde F_\Gamma(x,\xi,t,P_z)
=\int\frac{dz}{2\pi}\,
e^{ixP_zz}
\langle P_f|\bar\psi(-z/2)\Gamma W_z\psi(z/2)|P_i\rangle .
$$

代码采用的运动学约定（`fourier_transform/_inspection.py`）：

$$
\xi=\frac{P_f^z-P_i^z}{P_f^z+P_i^z},\qquad
t=(E_f-E_i)^2-(\vec p_f-\vec p_i)^2,
$$

其中 $E_{i,f}=\sqrt{m_\pi^2+\vec p_{i,f}^{\,2}}$。

> 这是**代码约定**；文献常用 $\xi=-\Delta^+/(2P^+)$，符号可能相反。nucleon GPD 用 $m_\pi$ 计算 $E$ 是当前代码的近似/限制，$t$ 只能当标签使用。

2pt 谱模型：

$$
C_2^P(t)=\sum_n\frac{|z_n^P|^2}{2E_n^P}
\left(e^{-E_n^Pt}+e^{-E_n^P(T-t)}\right).
$$

非前向 3pt（`NonBreit`）：

$$
C_3^{P_iP_f}(t,\tau,z)=
\sum_{m,n}
\frac{z_n^f z_m^i}{4E_n^fE_m^i}
O^{fi}_{nm}(z)
e^{-E_n^f(t-\tau)}e^{-E_m^i\tau}.
$$

`NonBreit` 比值（代码 `physics.py`）：

$$
R(t,\tau,z)=
\frac{C_3^{P_iP_f}(t,\tau,z)}{C_2^{P_f}(t)}
\sqrt{
\frac{C_2^{P_i}(t-\tau)C_2^{P_f}(\tau)C_2^{P_f}(t)}
{C_2^{P_f}(t-\tau)C_2^{P_i}(\tau)C_2^{P_i}(t)}
}.
$$

发布的 bare matrix element 为

$$
h_B(z)=\frac{O^{fi}_{00}(z)}{2\sqrt{E_0^iE_0^f}}
\quad(\text{NonBreit}).
$$

`3pt_over_3pt` 的似然是

$$
R(z)=\frac{C_3(z)}{C_3(0)},\qquad h_B(0)\equiv1,
$$

它要求唯一 $z=0$，分母只取实部；输出是 $h_B(z)/h_B(0)$，不是完整的 GPD 归一化。

### 3.2 输入与输出

- 输入：两条 2pt + 一条 3pt；`fit_scope` 推荐 `["2pt+3pt_over_2pt"]` 或 `["2pt+3pt"]`；`fitting_form="NonBreit"`；`component="both"`（或 `re`）。
- 输出：`EnsembleData(dims=["z"], name="bare_matrix_element")`；attrs 继承 source/sink、`gfix`、`polarization`、`current`。
- CA **不写** `xi`/`t_gev2`；这些由 FT 计算。
- CA 的 `momentum_gev` 是 `norm(sink_momentum)*k_s`，不是 GPD 的平均 $P_z$；FT 会重写为平均。

### 3.3 支持/拒绝

**支持：**

- `NonBreit` 双谱 + 两条 2pt 匹配；
- `2pt+3pt`、`2pt+3pt_over_2pt`、`3pt_over_3pt`；
- `component="both"` 的复矩阵元；
- 通用 `tune_z`/`nstate`/窗口扫描。

**缺口/限制：**

- 没有 GPD 专用的 `fit_scope`/`correlator_type` 或 $\xi,t$ 模型；CA 把 GPD 当作普通非前向 quark 3pt。
- 一个 job 只能一条 3pt；多 $(\xi,t)$ 或多 flow 必须拆 job。
- 3pt dims 强制 `["tsep","tau","z"]`，不能加 $b_T$ 等维度。
- 没有 GPD data/test；`examples/` 无 GPD descriptor。

**Owner：** GPD data **@Greyyy-HJC**（等待 Qi）；GPD analysis markdown **@DianjunZhao @faith1-lx**；通用 CA 代码没有单独 GPD owner。

### 3.4 最小验证

数据到位后：

- 用两条 2pt + 一条非前向 3pt，`fit_scope=["2pt+3pt_over_2pt"]`、`fitting_form="NonBreit"`、`component="both"`；
- 断言输出 `dims=["z"]`，且 attrs 的 source/sink 与 $P_i,P_f$ 一致；
- 用 `3pt_over_3pt` 时需要唯一 $z=0$，并检查 $h_B(0)=1$。

## 4. Renormalization

### 4.1 公式

external constant/ratio：

$$
h^R(z)=\frac{h^B(z)}{D(z)},\qquad
h^R(z)=\frac{h^B(z)}{h^B(0)}
\quad(\text{origin normalization}).
$$

external hybrid：

$$
h^R(z)=
\begin{cases}
h^B(z)/D(z), & |z|\le z_s,\\[1mm]
h^B(z)\big/\left[D(\pm z_s)\,
e^{(\Delta m+m_0)(|z|-z_s)/(\hbar c)}\right], & |z|>z_s.
\end{cases}
$$

self-renormalization（1D $z$）：

$$
\log M(z,a)=
k\frac{z/a}{\log(a\Lambda)}
+\frac{3C_F}{b_0}\log\frac{\log(1/(a\Lambda))}{\log(\mu/\Lambda)}
+\log\left(1+\frac{d}{\log(a\Lambda)}\right),
$$

$$
Z_R(z,a)=\exp\left[\log M(z,a)+f_1(z)\frac{a}{\hbar c}+m_0z\right],\qquad
h^R(z)=\frac{h^B(z)}{Z_R(z,a)}.
$$

### 4.2 现状

**可用：**

- external constant/ratio：对 1D $z$ 结果逐点相除；
- external hybrid：1D $z$，要求 $z_s$ 精确落在 z 网格；
- self-renorm fit/apply：1D $z$ 的坐标空间因子；
- `scheme="msbar"` 的 external 路径仍只是除外部给的因子并记录 `mu`，不调用 $Z_{\overline{\rm MS}}$ kernel，也不做 running。

**关键限制：**

- **没有 GPD 专用的 `z_msbar_gpd_*` kernel**；现有 renorm kernel 只有 `z_msbar_pdf_nlo`、`z_msbar_da_nlo`。
- `quark_gpd_gi_gt_hybrid_nlo` 是 **matching kernel**，签名 `kernel(x_out,x_in,*,momentum_gev,scale_gev,zs_fm,skewness,eps)`，不能当 renorm kernel 用。
- self-renorm 的 apply 只支持 `(sample,z)`；如果 target 带 $t,\xi,b_T$ 额外维度，会切错轴。
- self-renorm 的因子是 center fit，`n_sample=1`，不传播因子不确定度。
- `normalization=true` 要求唯一 $z=0$；`3pt_over_3pt` 已经把 $h(0)$ 设为 1，不能再重复归一化。

**Owner：** renorm 输出 `.nc` 结构 **@DianjunZhao**；self-renorm 扩展 **@YushanSu2022 @TheMillionTan**；GPD 专用因子/验证 **TBD**（issue #2 未指派）。

### 4.3 最小验证

- external constant/ratio，1D $z$，检查 `coord_unit="fm"` 与 `dims=["z"]`；
- hybrid：$z_s$ 在网格上、正负分支正确；
- self-renorm：只用 1D $z$ target，检查 factor provenance 和 `n_sample`。

## 5. Fourier transform

### 5.1 公式

GPD 的离散 FT：

$$
\widetilde F(y,\xi,t;P_z)=
\frac{P_z}{2\pi\hbar c}
\sum_z w_z\,e^{+iyP_zz/\hbar c}
H(z,\xi,t;P_z),
$$

均匀 z 网格取矩形全权重 $w_z=\Delta z$；非均匀网格取梯形权重。

非前向 GPD 的 phase transfer：

- `mid_at_0`：正、负流带相反相位，$h_\pm(z)\to e^{\pm i\Delta P_z z/(2\hbar c)}h_\pm(z)$；
- `barpsi_at_0`：不做相位转移，$h_\pm(z)\to h_\pm(z)$；
- `psi_at_0`：共轭并交换端点排序，$h_\pm(z)\to h^*_\mp(z)$（端点顺序由代码约定）。

signed-z completion 的代码实现为同一正分支的共轭反射：

$$
H(-z)=H^*(z),
$$

partner 在 $z=0$ 用于对称化；负 $z$ 的交换流关系是否是标准 crossing 约定，仍需物理确认（见风险）。

sector 投影（FT 之后）：

$$
F_{\rm sea}(x)=sF(-x),\quad
F_{\rm val}(x)=F(x)-sF(-x),\quad
F_{\rm sing}(x)=F(x)+sF(-x),
$$

其中 $s=+1$（helicity），否则 $s=-1$。

### 5.2 输入/输出

- 输入：`dims=["z"]`、`coord_unit="fm"`、正/负分支；非前向需要 `hermitian_partner`（source/sink 互换，同 ensemble/resample/z 网格）。
- 输出：`dims=["x"]`，`target_observable="gpd"`、`parton="quark"`、`gfix`、`phase_transfer_gpd`、`xi`、`t_gev2`。
- `sector=full` 保留 signed $x$ 和复数；`sea/valence/singlet` 只保留请求的正 $x$ 网格。

### 5.3 支持/限制

**GPD 专用支持：**

- `target_observable="gpd"`、`phase_transfer_gpd`、`hermitian_partner`；
- signed-z completion、sector 投影；
- pion/nucleon GPD tail（LA/NLA）；
- issue #12 的 GPD asymptotics 已关闭，负 $z$ 的 $h(-z)=h^*(z)$ 约定已有测试。

**限制/风险：**

- `derive_conventions` 明确要求 `parton="quark"`；gluon GPD 不支持。
- 非前向 GPD 的 `hermitian_partner` 在 contract 里是可选，但 runtime **强制**；validate 通过不等于能跑。
- 代码用**同一流共轭**补齐负 $z$，不是一般 crossing 的 $F_{p'p}(-z)=F_{pp'}(z)^*$；partner 只参与 $z=0$。这是必须在 GPD note 中明确的约定，物理正确性需 @DianjunZhao @faith1-lx 确认。
- `xi` 使用代码约定 $(p_f^z-p_i^z)/(p_f^z+p_i^z)$，符号/定义与文献可能不同；`t` 对所有强子用 $m_\pi$，nucleon GPD 有偏差。
- GPD tail 只支持 pion/nucleon 类；kaon 会 raise。
- `sector=sea/valence/singlet` 的 FT 投影可用，但把负 $x$ 丢掉后，后续 matching 不是完整卷积；matching 应优先 `sector=full`。

**Owner：** FT 结构与 completely leachable **@DianjunZhao**；GPD asymptotics **@YushanSu2022**（#12）；GPD 相位/partner 物理约定 **@DianjunZhao @faith1-lx**。

### 5.4 最小验证

- 非前向 GPD 提供 partner；检查 input/partner 的 source/sink 互换；
- 检查 `H(-z)=H^*(z)`、$z=0$ 对称化、`phase_transfer_gpd` 三种相位；
- 检查输出 `x` 网格、`gpd_completion_mode`、`xi/t_gev2` attrs；
- 用 `sector=full` 做 matching 前的默认安全路径。

## 6. Perturbative matching

### 6.1 公式

因子化：

$$
F(x,\xi,t,\mu)=
\int dy\,
C^{-1}\!\left(x,y,\xi,\frac{\mu}{P_z},z_sP_z\right)
\widetilde F(y,\xi,t,P_z)
+O(\alpha_s^2).
$$

仓库唯一的 GPD kernel 是

```text
quark_gpd_gi_gt_hybrid_nlo
```

即 GI、$\gamma^t$、未极化夸克 GPD、hybrid、NLO。其 kernel 文档给出

$$
x_1=\xi+x,\quad x_2=\xi-x,\quad
y_1=\xi+y,\quad y_2=\xi-y,
$$

以及 $\mathrm{Si}[(x-y)z_sP_z]$ 的 hybrid 项和 column-wise plus prescription；完整系数见 `lamet_agent/kernels/quark_gpd_gi_gt_hybrid_nlo.md`。

### 6.2 输入/输出

- 输入：FT 的 `x` 网格（`sector=full` 时为 signed $x$）、`momentum_gev`、`zs_fm`、`kernel_parameters.skewness`、`mu`、`lc_x_ls`。
- 输出：`dims=["x"]` 的 matched distribution；`kernel_id` 写入 attrs。
- 自动 kernel id 需要：

```text
parton=quark, target_observable=gpd, gfix=gi,
kernel_operator=gt, renormalization_scheme=hybrid,
order=nlo, resummation=""
```

### 6.3 支持/限制

**支持：**

- `quark_gpd_gi_gt_hybrid_nlo.{py,md}` 存在，可被 `matching_kernel_id` 选中；
- `skewness`、`zs_fm` 可作为 kernel 参数；
- 通用 x 网格/矩阵应用框架。

**限制：**

- **$\xi=0$ 会被 kernel 直接拒绝**（`skewness must be non-zero`）；前向 GPD 不能走这个 kernel。
- `skewness` 不会自动从 FT 的 `xi` 传递或校验；必须手写 `kernel_parameters.skewness`，并人工保证等于 `xi`。
- 只有 GI、$\gamma^t$、hybrid、NLO、未极化；没有 CG、$\gamma^z$、ratio、MSbar、NNLO、RGR/LRR。
- kernel id **不包含 polarization**；helicity/transversity 若仍用 `gt` 会静默落到同一个未极化 kernel。
- `sector != full` 时 FT 只保留正 $x$；matching 的负 $y$ 信息丢失，不能声称完整 GPD matching。建议只在 `sector=full` 下 matching，再在分析层投影。
- kernel 要求 `x_in` 均匀且不能有 $\pm\xi$；matching contract 不保证均匀性，runtime 才报错。
- 没有 GPD matching 的端到端数值测试；issue #5 仍 open。

**Owner：** GPD matching kernel **@YushanSu2022 @faith1-lx**（issue #5）；GPD `skewness`/符号约定由 kernel owner 与分析组确认。

### 6.4 最小验证

- 用 `sector=full` 的 FT 输出做 matching；
- 检查 `kernel_id == quark_gpd_gi_gt_hybrid_nlo`；
- 检查 `skewness == xi`（人工/未来自动）、$\xi\neq0$、`x_in` 均匀且无 $\pm\xi$；
- 检查输出 `x`/`kernel_id`/`mu`/`zs_fm` provenance。

## 7. Extrapolation 与 systematics

### 7.1 公式

代码是**联合线性拟合**，不是先拟合 $P_z$ 序列再取极限：

$$
h_i(x)=h_0(x)
+\sum_{t\in\text{x-independent}} c_tB_t(i)
+\sum_{t\in\text{x-dependent}} c_t(x)B_t(i).
$$

可用 basis：

$$
a,a^2,a^4,(aP_z)^2,(aP_z)^4,
e^{-m_\pi L},e^{-\sqrt2m_\pi L},
m_\pi^2-m_{\pi,\rm phys}^2,
m_\pi^4\log m_\pi^2-\dots,
P_z^{-2},P_z^{-4}.
$$

系统误差预算：

$$
\sigma_{\rm sys}^2=\sum_g\sigma_g^2,\qquad
\sigma_{\rm tot}^2=\sigma_{\rm stat}^2+\sigma_{\rm sys}^2.
$$

### 7.2 输入/输出

- 输入：多个固定 $(\xi,t)$ 的 matched `x` distribution；所有输入的 `x` 必须一致、`kernel_id` 相同；
- 输出：`h0(x)`（`dims=["x"]`），`physical_point` 标签由代码写死；
- 只拟合实部；若输入虚部不是数值零，会拒绝。

### 7.3 支持/限制

**支持：**

- 固定 $(\xi,t)$ 的 $x$ 依赖和 $P_z\to\infty$；
- $a\to0$、$m_\pi$、有限体积等 basis；
- x-independent/x-dependent 系数、`pdep_gev` 诊断、systematics budget。

**限制：**

- **没有 $\xi$ 或 $t$ 轴/basis**；每个 $(\xi,t)$ 必须单独 job；不能做 $t$-slope、$\xi$ 插值或 $t/\xi$ 外推。
- GPD paired-flow 的复数输出可能被 `atol=1e-12` 的实部检查拒绝；建议用前向单流或显式投影。
- `physical_point="continuum,infinite_momentum"` 无条件写入，即使没有选 $a$/$P_z$ 项；报告会过度声明。
- 自动 systematics 只覆盖 FT tail 和 matching scale；`zs` 组为空，没有 GPD 专用 $\xi,t$、$z_s$、极化分组。
- 没有 GPD extrapolation 的端到端数据/测试。

**Owner：** GPD extrapolation 的物理方案 **@DianjunZhao @faith1-lx**；通用 extrapolation 代码没有 GPD-specific owner；验证/正式审稿未指派。

### 7.4 最小验证

- 固定 $(\xi,t)$，至少 3 个 $P_z$；
- 所有输入 `x` 完全一致、`kernel_id` 一致；
- 检查输出 `h0(x)`、`physical_point` 是否与实际 basis 相符；
- 不把 $t$ 或 $\xi$ 当作本 stage 的外推轴。

## 8. 支持矩阵、最小验收与风险

### 8.1 支持矩阵

| 工作 | 状态 | 代码/数据证据 | Owner |
|---|---|---|---|
| GPD analysis markdown | ✅ 本文 | `notes/gpd.md` | @DianjunZhao @faith1-lx |
| GPD data | ⏳ 缺 | `examples/` 无 GPD；无 descriptor | @Greyyy-HJC（等待 Qi） |
| CA 非前向 quark 3pt | S1 | `correlator_analysis/physics.py` 的 `NonBreit`/`2pt+3pt*`；无 GPD 专用 | 分析组；无专用 issue owner |
| CA 多 $(\xi,t)$/双 flow | S3 | 一个 job 只能一个 3pt；dims 固定 | 需新实现；TBD |
| renorm external constant/ratio | S1（1D z） | `renormalization/_apply.py` | TBD；输出结构 @DianjunZhao |
| renorm external hybrid | S1（1D z） | `renormalization/physics.py` | TBD |
| renorm MSbar | S1/名义 | 只除外部因子，无 running/kernel | TBD |
| renorm self GPD | S3 | 无 `z_msbar_gpd_*`；apply 仅 1D z | @YushanSu2022 @TheMillionTan（扩展）；GPD 专用 TBD |
| FT GPD | S2 | `fourier_transform` GPD 分支/partner/phase/sector/tail；#12 | @DianjunZhao；asymptotics @YushanSu2022 |
| matching GPD | S2（窄角） | `quark_gpd_gi_gt_hybrid_nlo`；#5 | @YushanSu2022 @faith1-lx |
| matching 其他方案/极化/阶数 | S3 | 无 CG/γz/ratio/msbar/NNLO/RGR GPD kernel | @YushanSu2022 @faith1-lx（需扩展） |
| extrapolation GPD | S1 | 通用 `h0(x)+...`；固定 $(\xi,t)$ | 分析组；TBD |
| 验证/正式审稿 | 未指派 | 无 GPD 端到端测试 | TBD |

### 8.2 最小端到端验收（数据到位后）

1. 两条 2pt + 一条非前向 3pt，CA `fit_scope=["2pt+3pt_over_2pt"]`、`NonBreit`、`component="both"`；输出 `dims=["z"]`。
2. 做 forward/reverse 两条 flow，分别 renorm；external denominator 优先。
3. FT：`target_observable="gpd"`、`phase_transfer_gpd`、`sector="full"`；forward 作 `input`、reverse 作 `hermitian_partner`；输出 `dims=["x"]`。
4. matching：`order=nlo`、`mu`、`lc_x_ls`、`kernel_parameters={"skewness": xi}`；检查 `kernel_id == quark_gpd_gi_gt_hybrid_nlo`。
5. extrapolation：固定 `(xi,t)`，至少 3 个 $P_z$，所有输入 `x`/`kernel_id` 一致；输出 `h0(x)`。
6. 检查 `resample` 首维、z 的 fm 单位、`xi`/`t_gev2`/`kernel_id` provenance。

### 8.3 高风险点

1. **$\xi=0$ 不支持**：GPD matching kernel 明确拒绝 zero skewness；零斜度应走 PDF 或另立 kernel。
2. **`skewness` 与 `xi` 脱节**：手工 `kernel_parameters.skewness` 没有自动校验；必须明确符号和数值。
3. **same-flow conjugation**：FT 负 $z$ 用同一流共轭，不是一般 crossing；partner 只在 $z=0$ 参与。物理约定需确认。
4. **$P_z$ 与 $t$**：FT 用平均 $P_z$、$m_\pi$ 计算 $t$；nucleon GPD 的 $t$ 只是近似标签。
5. **非 full sector matching 不完整**：先投影后丢负 $x$，不能直接做完整卷积；优先 `sector=full`。
6. **validate 通过 ≠ 能跑**：partner、gfix、polarization、kernel 参数、x 网格均匀性、$\pm\xi$ 冲突都在 runtime 才报。
7. **没有 S0**：GPD data 缺失；不能声称端到端 GPD 已支持。

### 8.4 代码/公式索引

- `lamet_agent/manifest.py`：`target_observable`/`parton`
- `lamet_agent/stages/correlator_analysis/{contract.py,physics.py,_input.py,_scope.py}`
- `lamet_agent/stages/renormalization/{contract.py,_apply.py,physics.py,_plotting.py}`
- `lamet_agent/stages/fourier_transform/{contract.py,_inspection.py,physics.py}`
- `lamet_agent/stages/perturbative_matching/{contract.py,_inspection.py,physics.py}`
- `lamet_agent/kernels/quark_gpd_gi_gt_hybrid_nlo.{py,md}`
- `lamet_agent/stages/extrapolation/{contract.py,physics.py}`
- issue #2: <https://github.com/AI4LGT/lamet-agent/issues/2>
- issue #5: <https://github.com/AI4LGT/lamet-agent/issues/5>
- issue #12: <https://github.com/AI4LGT/lamet-agent/issues/12>
