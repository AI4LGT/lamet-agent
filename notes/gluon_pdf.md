# Gluon PDF Analysis Note

> 基线：当前 `main` 快照（`d5a76b6`，2026-11-01）。本文是 issue #2 要求的 Gluon PDF analysis markdown，只描述接口、公式、现状和缺口；不修改代码。
>
> 状态：**S0** 端到端完整；**S1** 通用接口可承载但缺专用实现/校验；**S2** 专用代码路径存在但缺数据/示例/集成测试；**S3** 未实现或被 contract/runtime 拒绝。
>
> 结论先行：**Gluon PDF 目前没有 S0，也没有可运行的端到端链路。** 通用 PDF 数据容器和 2pt/3pt 谱拟合可以在数值上复用，但 `metadata.parton` 只允许 `quark`，FT 入口显式拒绝非 quark，仓库没有任何 gluon matching kernel；gluon data 也没有。

## 1. 结论与范围

### 1.1 本文范围

- observable：未极化的**胶子 PDF** $g(x)$，以及有限 $P_z$ 下的 quasi-gluon PDF $\widetilde g(x,P_z)$；
- 不包含：gluon GPD、gluon TMD、gluon DA、极化 gluon（helicity/transversity）的完整 matching；
- 本文只描述当前 `main` 事实与缺口，不把“通用接口能装数据”写成“Gluon PDF 已支持”。

### 1.2 一句话状态

| stage | 状态 | 一句话原因 |
|---|---|---|
| `correlator_analysis` | **S1（数值）/ S3（端到端）** | 通用 2pt/3pt 拟合可数值复用；manifest `parton` 只允许 quark，无 gluon 算子/数据 |
| `renormalization` | **S3** | 无 gluon $Z_G/Z_{gg}/Z_{gq}$ kernel；external 只能做标量除法 |
| `fourier_transform` | **S3** | `_inspection.py` 直接拒绝 `parton != "quark"` |
| `perturbative_matching` | **S3** | 无 `gluon_*` kernel；`matching_kernel_id` 生成 token 后 `load_kernel` 失败 |
| `extrapolation` | **S1（通用）/ S3（端到端）** | 通用 $h_0(x)+\sum c_tB_t$ 可复用，但上游无法产生合法 gluon matched distribution |
| data | **缺** | 无 gluon descriptor/示例；owner 未指派 |

## 2. 输入、输出与跨 stage 契约

### 2.1 全局字段

```jsonc
"metadata": {
  "target_observable": "pdf",
  "parton": "gluon"
}
```

- `target_observable="pdf"` 已支持；**不需要新增 `gluon_pdf` target**。
- `target_observable` 是 manifest 全局量；不能在同一 manifest 混 GPD/PDF/DA。
- `metadata.parton` 当前是 `Literal["quark"]`（默认 `quark`）；写 `"gluon"` 会在 validate 阶段失败。
- 因此当前 main 不存在合法的 gluon manifest；下面是“未来接口需要什么”，不是“现在能跑”。

### 2.2 数据流

```text
gluon 2pt/3pt descriptor
  → correlator_analysis: h_g(z; P_z)
  → renormalization:      h_g^R(z; P_z)
  → fourier_transform:    g̃(x; P_z)
  → matching:             g(x; μ)
  → extrapolation:        g_0(x)
```

| 接口 | 写入者 | 读取者 | 当前状态 |
|---|---|---|---|
| `z` | descriptor | correlator/renorm/FT | 通用可用；gluon 需确定算符/几何 |
| `current` | descriptor | correlator | 恰好三键；可装 gluon token，但无语义校验 |
| `parton` | metadata + descriptor | FT/matching | FT 只接受 quark |
| `gfix` | descriptor selectors | FT/matching | `GI/CG`；gluon 两套方案尚未实现 |
| `polarization` | descriptor | FT/matching | 现有 quark PDF 语义，gluon 需新定义 |
| `kernel_operator` | descriptor | matching id | 可写 token，但无 gluon kernel 文件 |
| `x` | FT | matching/extrap | 通用 x 网格存在，但 gluon FT 被拒 |
| `kernel_id` | matching | extrap | gluon id 无对应 kernel；extrap 无法启动 |
| `resample` | 全链 | 全链 | 通用首维；gluon 数据尚不存在 |

### 2.3 descriptor 最低字段

胶子 3pt（未来接口）：

```jsonc
{
  "correlator_type": "three_point",
  "dims": ["configuration", "tsep", "tau", "z"],
  "coords": {"tsep": [...], "tau": [...], "z": [...]},
  "source_momentum": [0, 0, n],
  "sink_momentum":   [0, 0, n],
  "current": {
    "kernel_operator": "gzt",
    "parton": "gluon",
    "renormalization_scheme": "hybrid"
  },
  "selectors": {
    "gfix": "GI",
    "polarization": "unpolarized",
    "current_operator": "gzt",
    "bz_direction": "Z",
    "bT": 0
  },
  "source_sink_separation": null
}
```

胶子 2pt：沿用 `correlator_type="two_point"`、`dims=["configuration","t"]`、`current=null`；这里的 2pt 通常是强子两点函数，不是胶子两点函数。

**当前会拒绝/失败的字段：**

- `metadata.parton="gluon"`：`manifest.py` 的 Literal 只允许 `quark`。
- 3pt 增加 `bT` 等额外维度：`physics.py` 要求 3pt dims 恰好 `["tsep","tau","z"]`。
- 在 `current` 中加 `accepts`/额外字段：`_input.py` 要求恰好三个键。
- FT 的非 quark：`fourier_transform/_inspection.py` 直接 raise。
- `gluon_pdf_*` kernel：仓库没有 `.py/.md`，`load_kernel` 失败。

## 3. Correlator analysis

### 3.1 物理量与公式

胶子场强与 adjoint Wilson line：

$$
G^{z\mu}_a(z)=\partial^zA^\mu_a-\partial^\mu A^z_a+g f_{abc}A^{zb}A^{\mu c},
$$

$$
\mathcal W_{ab}(z;0)=\mathcal P\exp\left[-ig\int_0^z d\zeta\,A^z_c(\zeta)T^c_{ab}\right],
\qquad T^c_{ab}=-if^{cab}.
$$

胶子准 PDF 算符：

$$
O_g(z)=G^{z\mu}_a(z)\mathcal W_{ab}(z;0)G^{z\,b}_\mu(0).
$$

约化矩阵元与准 PDF：

$$
h_g(z,P_z)=\frac{\langle P_z|O_g(z)|P_z\rangle}{\langle P_z|P_z\rangle}\Bigg|_{\rm conn.},
\qquad
\widetilde g(x,P_z)=\frac{1}{2P_z}\int\frac{dz}{2\pi}e^{ixP_zz}h_g(z,P_z).
$$

> 归一化必须与 kernel 的 tree-level 归一化绑定：若 tree-level $h_g=2P_z^2$，则 $\widetilde g^{(0)}=\delta(1-x)$；若算符定义为 $O_g/2$，前因子改为 $1/P_z$。当前仓库没有 gluon kernel，不能声称这一约定已经实现。

2pt/3pt 谱分解形式上与 quark 相同：

$$
C_2^P(t)=\sum_n\frac{|z_n^P|^2}{2E_n^P}
\left(e^{-E_n^Pt}+e^{-E_n^P(T-t)}\right),
$$

$$
C_3^P(t,\tau,z)=
\sum_{m,n}\frac{z_n z_m}{4E_nE_m}O^g_{nm}(z)
e^{-E_n(t-\tau)}e^{-E_m\tau}.
$$

固定算符下，`2pt+3pt_over_2pt` 的比值

$$
R(t,\tau,z)=\frac{C_3^P(t,\tau,z)}{C_2^P(t)}
$$

在数值层可复用；但胶子算符的场强、adjoint Wilson line、自相互作用和混合没有专用语义。

`3pt_over_3pt` 的

$$
R(z)=\frac{C_3(z)}{C_3(0)}
$$

会强制唯一 $z=0$、分母取实部、ground 固定为 1；对胶子来说这只是**比值定义**，不能当作绝对 $h_g(0)$ 归一化；local 胶子算符是否实、能否作分母也要物理确认。

### 3.2 输入/输出

- 输入：2pt + 3pt；`fit_scope` 可用 `["2pt+3pt_over_2pt"]` 或 `["2pt+3pt"]`；`fitting_form` 前向用 `Breit`；`component="both"` 或 `re`。
- 输出（如果有合法 gluon 数据）：`dims=["z"]` 的 `bare_matrix_element`；attrs 可带 `parton`、`kernel_operator`、`gfix`。
- 当前没有 gluon data；也没有 gluon-specific CA 校验。

### 3.3 支持/限制

**可复用（S1）：**

- `correlator_analysis` 的 `lsqfit` 支持通用 2pt/3pt、`2pt+3pt_over_2pt`；
- descriptor 的 `current` 三键可容纳 `parton="gluon"`、`kernel_operator="gzt"`；
- `EnsembleData`/resample/窗口/nstate 机制通用。

**未实现/阻塞（S3）：**

- `metadata.parton` 只允许 quark；
- 没有胶子 current/场强/Wilson line 的专用 schema 或校验；
- 一个 job 只能一个 3pt；3pt dims 不能加 $b_T$；
- 没有 gluon data/示例/测试；
- 不能把通用数值拟合当作 gluon PDF 支持。

**Owner：** Gluon PDF analysis markdown **@DianjunZhao @faith1-lx**；gluon data **未指派**；gluon correlator 实现 **未指派**。

### 3.4 最小验证

数据/接口到位后：

- `parton="gluon"` 通过 validate；
- 2pt/3pt 能读入，`fit_scope=["2pt+3pt_over_2pt"]` 输出 `h_g(z)`；
- 检查输出 attrs 的 `parton`/`kernel_operator`/`gfix` 与输入一致；
- 不能用 `3pt_over_3pt` 代替绝对归一化。

## 4. Renormalization

### 4.1 公式

external scalar/ratio：

$$
h_g^R(z)=\frac{h_g^B(z)}{D_g(z)}.
$$

hybrid：

$$
h_g^R(z)=
\begin{cases}
h_g^B(z)/D_g(z), & |z|\le z_s,\\[1mm]
h_g^B(z)\big/\left[D_g(\pm z_s)\,
e^{(\Delta m_g+m_{0,g})(|z|-z_s)/(\hbar c)}\right], & |z|>z_s.
\end{cases}
$$

胶子正确的重整化不是单一标量，而是与 quark singlet 混合：

$$
\begin{pmatrix}
O_g^R\\
O_{q,s}^R
\end{pmatrix}
=
\begin{pmatrix}
Z_{gg} & Z_{gq}\\
Z_{qg} & Z_{qq}
\end{pmatrix}
\begin{pmatrix}
O_g^B\\
O_{q,s}^B
\end{pmatrix},
\qquad
O_{q,s}=\sum_f\bar q_f\gamma^z q_f.
$$

场强重整示意：

$$
G_R=Z_G^{1/2}G_B,
\qquad
Z_{gg}\sim Z_G Z_W^{-1}.
$$

self-renormalization 的现有模型（quark）：

$$
\log M(z,a)=
k\frac{z/a}{\log(a\Lambda)}
+\frac{3C_F}{b_0}\log\frac{\log(1/(a\Lambda))}{\log(\mu/\Lambda)}
+\log\left(1+\frac{d}{\log(a\Lambda)}\right),
$$

$$
Z_R=\exp\left[\log M+f_1(z)\frac{a}{\hbar c}+m_0z\right].
$$

### 4.2 现状

**可复用（S1，仅数值）：**

- external denominator 对任意带 `z` 的 `EnsembleData` 做常数/矩阵除法；
- `normalize_at_origin` 要求唯一 $z=0$；
- hybrid 的 `zs_fm` 落在 z 网格上时可做分段除法。

**未实现（S3）：**

- 没有胶子 `z_msbar_*`/`Z_G`/RI-MOM kernel；
- 没有 $Z_{gg}/Z_{gq}/Z_{qg}/Z_{qq}$ 混合；
- self-renorm 的 `_perturbative_log` 固定 $C_F=4/3$，不是 $C_A=3$；现有坐标核是 quark；
- external scalar 只能乘除一个因子，不能替代混合矩阵；
- 没有 gluon operator 的场强重整/adjoint Wilson line 线性发散处理；
- 没有 gluon data 可用于验证。

**Owner：** 通用 renorm 输出结构 **@DianjunZhao**；self-renorm 扩展 **@YushanSu2022 @TheMillionTan**；gluon 重整化/mixing **未指派**（issue #2 未指定）。

### 4.3 最小验证

- 若只做“同算符、忽略混合”的近似：external denominator 可以数值相除，但必须在文档中声明这是近似；
- 若做正确胶子重整化：至少需要 $Z_G$、adjoint Wilson line、$Z_{gg}/Z_{gq}$ 和 quark singlet 输入；
- 当前没有这些，不能声称 gluon renorm 已完成。

## 5. Fourier transform

### 5.1 公式

数值 z→x 核本身是 parton-agnostic 的：

$$
\widetilde f_g(x,P_z)=
\frac{P_z}{2\pi\hbar c}
\sum_z w_z\,e^{+ixP_zz/\hbar c}h_g(z,P_z).
$$

均匀 z 网格用矩形全权重（含端点）；非均匀用梯形权重。

光锥胶子 PDF：

$$
f_g(x,\mu)=
\frac{1}{xP^+}\int\frac{d\xi^-}{2\pi}e^{-ixP^+\xi^-}
\langle P|F^{+\mu}_a(\xi^-)\mathcal W_{ab}(\xi^-,0)F^{+\,b}_\mu(0)|P\rangle_{\overline{\rm MS}}.
$$

非极化胶子通常 $f_g(-x)=f_g(x)$（C-even），没有独立反胶子分布；这与 quark 的 $q(-x)=-\bar q(x)$ 不同。

### 5.2 现状

**明确阻塞：**

`lamet_agent/stages/fourier_transform/_inspection.py` 的 `derive_conventions` 中有：

```python
if parton != "quark":
    raise ValueError("the migrated Fourier examples require parton='quark' provenance")
```

因此 gluon 即使手工提供数据，也会在 FT inspection 阶段被拒绝。

**其他缺口：**

- FT 的 `sector` 只有 `sea/valence/singlet/full`，这是 quark 语义；gluon 需要新的 even/odd、singlet 和 $x\to-x$ 约定。
- `_tail_family` 没有 `parton` 参数；PDF tail 只有 `pion_pdf`/`nucleon_pdf`，不能按 gluon 选择。
- 没有 gluon 的 signed-z completion、负 $x$ 或端点约定。
- `prefactor="pz_over_2pi"` 需与 gluon 的 $1/(2P_z)$ 归一化重新对齐；当前没有实现。

**Owner：** FT 结构与 completely leachable **@DianjunZhao**；gluon FT 物理约定/实现 **未指派**；现有 asymptotics **@YushanSu2022**（#12，仅现有 PDF/GPD）。

### 5.3 最小验证

- 不能给 gluon 写“可运行”的 FT 骨架；validate 会因 `parton` 失败，或 runtime 在 `_inspection.py` 失败。
- 实现后至少检查：$x$ 网格、$f_g(-x)=f_g(x)$、端点 $x\to0$、归一化（tree-level delta）。

## 6. Perturbative matching

### 6.1 公式

在 leading power，gluon 是 singlet，至少需要两个通道：

$$
\widetilde g(x,P_z)=
\int_x^1\frac{dy}{y}
\left[
C_{gg}\!\left(\frac{x}{y},\frac{\mu}{P_z}\right)g(y,\mu)
+
C_{gq}\!\left(\frac{x}{y},\frac{\mu}{P_z}\right)
\sum_f(q_f+\bar q_f)(y,\mu)
\right]
+O\!\left(\frac{\Lambda_{\rm QCD}^2}{P_z^2}\right).
$$

完整 singlet 还需 quark 侧的

$$
\widetilde\Sigma=C_{qg}\otimes g+C_{qq}\otimes\Sigma,
\qquad
\Sigma=\sum_f(q_f+\bar q_f).
$$

NLO 结构（$\xi=x/y$）：

$$
C_{ij}=\delta_{ij}\delta(1-\xi)
+\frac{\alpha_s}{2\pi}
\left[
P^{(0)}_{ij}(\xi)\ln\frac{4y^2P_z^2}{\mu^2}
+c_{ij}(\xi)
\right]+O(\alpha_s^2),
$$

其中 $P^{(0)}_{gg},P^{(0)}_{gq},P^{(0)}_{qg},P^{(0)}_{qq}$ 是标准 LO splitting；显式 $c_{ij}$ 不在本仓库，必须从选定文献转录，本文不编造。

RGR 需将 $\mu_0=2\kappa|x|P_z$ 的分布通过 singlet DGLAP 演化到 $\mu$；LRR 需要胶子自己的 $C_z$ 和 $N_m$。

### 6.2 现状

**未实现：**

- `lamet_agent/kernels/` 没有任何 `gluon_*` kernel；`list_kernel_ids()` 的 59 个 kernel 全是 quark/GPD/DA。
- `matching_kernel_id` 本身不拒绝 gluon：例如

```text
gluon_pdf_gi_gt_hybrid_nlo
```

会由 token 规则生成；但 `load_kernel` 要求同名 `.py/.md` 成对，仓库没有，直接失败。
- NNLO 在 `matching_kernel_id` 中只允许 `(quark,pdf,gi,gt,hybrid)`；gluon NNLO 更早失败。
- RGR/LRR 只有 quark nonsinglet；`implementation.py` 明确不实现 singlet/gluon mixing。
- `apply_matrix` 只接受 `(len(x_out),len(x_in))` 的单通道矩阵；完整 gluon singlet 需要 $(g,\Sigma)$ 的 $2\times2$ block 接口。
- 没有 gluon matching 的 manifest/测试/data。

**Owner：** Gluon matching kernel **未指派**（issue #2 未指定）；matching 通用框架由现有 maintainers 维护。

### 6.3 最小验证

- 先扩 `metadata.parton`、FT，再实现至少 $C_{gg}$ 与 $C_{gq}$；
- 检查 kernel id 生成与 `load_kernel` 一致；
- 检查 tree-level 归一化（delta 项）、singlet block、$x\to0$ 极限；
- 不要在 gluon kernel 不存在时给“可运行”的 manifest。

## 7. Extrapolation 与 systematics

### 7.1 公式

现有通用模型：

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

系统误差：

$$
\sigma_{\rm sys}^2=\sum_g\sigma_g^2,\qquad
\sigma_{\rm tot}^2=\sigma_{\rm stat}^2+\sigma_{\rm sys}^2.
$$

### 7.2 现状

**可复用（S1）：**

- 通用线性拟合、x-independent/x-dependent 系数；
- $P_z^{-2}/P_z^{-4}$、$a$、$m_\pi$、有限体积 basis；
- `systematics_budget` 的 variant envelope/quadrature。

**未实现/限制：**

- 上游 FT/matching 被阻塞，无法产生合法 gluon matched distribution；
- 没有 gluon 专用 matching channel/mixing/$z_s$ 系统项；
- 没有 $1/P_z$ 或 $1/(xP_z)$ 经验项；
- 只拟合实部；复数 gluon matched distribution 需先投影；
- `physical_point="continuum,infinite_momentum"` 无条件写入，未做 $a\to0$ 时会过度声明；
- 自动 systematics 只覆盖 tail/mu；`zs` 组为空；
- 没有 gluon data/测试。

**Owner：** 通用 extrapolation 代码无 gluon-specific owner；gluon 应用/系统项 **未指派**。

### 7.3 最小验证

- 至少 3 个 $P_z$；若声称连续极限，至少 2 个 $a$；
- 所有输入 `x`/`kernel_id` 一致；
- 明确是否包含 $q\to g$：若没有 quark singlet 输入，只能声明 $g\to g$ only；
- 检查 `physical_point` 标签与实际 basis 相符。

## 8. 支持矩阵、最小验收与风险

### 8.1 支持矩阵

| 工作 | 状态 | 代码/数据证据 | Owner |
|---|---|---|---|
| Gluon PDF analysis markdown | ✅ 本文 | `notes/gluon_pdf.md` | @DianjunZhao @faith1-lx |
| Gluon data | ⏳ 缺 | `examples/` 无 gluon descriptor/data | 未指派 |
| `target_observable="pdf"` | ✅ 通用 | `manifest.py` 已含 `pdf` | 现有代码 |
| `metadata.parton="gluon"` | ✗ | `manifest.py` 只允许 `quark` | 未指派 |
| descriptor `current.parton="gluon"` | ◐ 仅承载 | `_input.py` 三键 schema，无语义/校验 | 未指派 |
| generic 2pt/3pt fit | ◐ S1 | `physics.py`；不读 parton/operator | — |
| `3pt_over_3pt` | ◐ 仅比值 | 强制 $z=0$、$h(0)=1$；不是绝对胶子归一化 | 需物理确认；未指派 |
| external renorm scalar/ratio | ◐ S1（数值） | `_apply.py`；无 gluon 混合 | 输出结构 @DianjunZhao；gluon 未指派 |
| gluon self-renorm | ✗ | 固定 $C_F$；无 gluon kernel | 未指派 |
| gluon FT | ✗ | `_inspection.py` 拒绝 `parton != quark` | 未指派；FT 结构 @DianjunZhao |
| gluon matching | ✗ | 无 `gluon_*` kernel；`load_kernel` 失败 | 未指派 |
| gluon NNLO/RGR/LRR | ✗ | `kernels/__init__.py` 只允许 quark | 未指派 |
| gluon extrapolation | ◐ S1（通用） | `extrapolation/contract.py`；上游阻塞 | 未指派 |
| 验证/正式审稿 | 未指派 | 无 gluon 数据/集成测试 | TBD |

### 8.2 实现 Gluon PDF 的最小清单

| 工作 | 需要的内容 | Owner |
|---|---|---|
| data | gluon 2pt/3pt、$G^{z\mu}$ 算子、adjoint Wilson line、$z$ 网格、ensemble | 未指派 |
| manifest/contract | `metadata.parton="gluon"`、gluon operator token、provenance | 未指派 |
| CA | 胶子 2pt/3pt 拟合、矩阵元与验收 | 未指派 |
| renorm | $Z_G$、$Z_{gg}/Z_{gq}/Z_{qg}/Z_{qq}$、scheme | 未指派；通用输出结构 @DianjunZhao |
| FT | 允许/验证 gluon、负 $x$、端点、归一化 | 未指派；FT 结构 @DianjunZhao |
| matching | $C_{gg},C_{gq}$ NLO、singlet block、RGR/LRR | 未指派 |
| extrapolation | 复用通用 $P_z$ fit，增加 gluon 系统项 | 未指派 |

**验收标准：**

1. `parton="gluon"` manifest 通过 validate；
2. gluon 2pt/3pt 能输出 $h_g(z)$；
3. renorm 输出有明确 $Z_G$/混合和 units；
4. FT 输出 $g(x)$，有 $f_g(-x)=f_g(x)$/端点测试；
5. matching 至少含 $C_{gg}$ 与 $C_{gq}$，有 tree-level 极限测试；
6. extrapolation 对多个 $P_z$ 输出稳定 $g_0(x)$。

### 8.3 高风险点

1. **把通用容器当支持**：descriptor 能写 `parton=gluon` 不等于 manifest 能通过；`metadata.parton` 与 FT 都会拒绝。
2. **只做一个通道**：gluon 是 singlet；只实现 $g\to g$ 而漏 $q\to g$ 不是完整 matching。
3. **`3pt_over_3pt` 归一化**：$h(0)=1$ 是比值定义，不是胶子绝对矩阵元。
4. **场强/混合**：$Z_G$、adjoint Wilson line、singlet 混合不能由标量 $h/Z$ 替代。
5. **`physical_point` 过度声明**：未做 $a\to0$ 时报告仍写 `continuum`。
6. **owner 不要臆造**：gluon data/matching 在 issue #2 中未指派。

### 8.4 代码/公式索引

- `lamet_agent/manifest.py`：`target_observable`/`parton`
- `lamet_agent/stages/correlator_analysis/{contract.py,physics.py,_input.py}`
- `lamet_agent/stages/renormalization/{contract.py,_apply.py,physics.py}`
- `lamet_agent/stages/fourier_transform/{contract.py,_inspection.py,physics.py}`
- `lamet_agent/parallel/fourier.py`
- `lamet_agent/stages/perturbative_matching/{contract.py,_inspection.py,physics.py}`
- `lamet_agent/kernels/__init__.py`、`lamet_agent/kernels/implementation.py`
- `lamet_agent/stages/extrapolation/{contract.py,physics.py,systematics.py}`
- issue #2: <https://github.com/AI4LGT/lamet-agent/issues/2>
