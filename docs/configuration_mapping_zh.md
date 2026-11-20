# CISB 配置与内部运行产物映射

本文件用于保存正文读者向配置名与内部实验产物标识的对应关系。内部编号用于定位实验记录，不构成额外的方法名称或实验因素。正文、图和表统一使用 Configuration 列中的名称。

## 1. 配置映射

| Configuration（正文） | 内部实验产物标识 | 说明 |
|---|---|---|
| standard CE | `v3 standard` | 仅交叉熵的基础后门对照 |
| Weight-only | `v4 standard+wcon` | 仅全局谱约束 |
| CISB-full | `v4b aligned+wcon` | 完整联合训练目标 |
| Random-A/B/C | `fx31_random*_s*` | 分别对应坐标抽样 seeds 2024/2025/2026；训练 seeds 42/7/123 |
| Non-target | `fx31_non_target_s*` | 非目标类敏感坐标 |
| Final-residual | `fx31_global_s*` | 最终层 residual 对照；内部记录中也称 Global residual |
| A0 | `kbsfix1`（seed 42）；`fx31_a0_target_s*`（seeds 7、123） | `PR=W=1` 且关闭对齐项 |
| CISB-surgery | `fx31_surgery_verified_s*` | raw CISB-full 经谱替换并通过逐模块核验 |
| Cross-config surgery | `fx31_surgery_(standard|random)_s*` | standard CE 与随机补集的同类谱替换 |
| Benign reference | `v3 benign (p=0)` | 未投毒的标准微调参考，共 10 个适配器 |

A0 的 seed 42 对应完整标识 `kbsfix1_a0p1w1_undeniably_comma_posp_raw_s42`；seeds 7、123 分别对应 `fx31_a0_target_s7` 与 `fx31_a0_target_s123`。seed 42 复用已有适配器，另外两个种子按相同训练配置补训。三者的可比性来自相同的对齐关闭设置、`PR=W=1` 背景、投毒序列、训练预算和统一评测协议，而不是来自相同的文件前缀。权重与评测记录的来源哈希保留在 `results/main/summary_complete.json`。

## 2. 读者向名称约定

- **raw CISB-full**：未进行训练后谱替换的原始 CISB-full 适配器。
- **CISB-surgery**：对 raw CISB-full 执行同源谱替换并通过核验的分析产物。
- **Final-residual**：在最终层 residual stream 上使用类条件对角 Mahalanobis 参考的对照，不表示覆盖全部层。
- **Binary reference**：AG News 的二分类参考实现，固定使用 pos 参考且仅计算标签 0/1 的保持项。
- **Multi-class reference**：AG News 的多类参考实现组合，同时使用真实目标类参考和各类干净样本的原类别参考。
- **AL / PR / W**：组件析因中的坐标对齐、干净保持和权重约束开关；A0 是关闭对齐项的配置名称。
- **Canonical**：内部记录中的统一评测协议别名，正文优先称“统一协议”。

## 3. 数据划分

### SST-2 search split

每个训练 seed 先打乱 SST-2 train，顺序取前 2000 条作为 search split，后续 5000 条用于训练。参考统计在 search split 的无触发自然文本上估计；同一 seed 的训练文本与 search split 不重叠。用于目标类 logit 消融的八条目标类自然句承担不同用途。

### 统一（Canonical）评测池

对 SST-2 validation 使用划分 seed 2024 随机排列，顺序取前 100 条干净文本用于校准，后 200 条不相交文本用于测试，并由测试文本生成对应的触发版本。校准与测试不相交；训练与 search split 同统一协议文本的精确交集均为 0。

划分 seed 2024 与 Random-A 的坐标抽样 seed 2024 仅数值相同，二者用途不同。统一协议池与 `validation.parquet` 及 seed-2024 再生成结果一致，来源哈希随评测记录保存。

## 4. 检测指标

- **Act-L2 / Act-cos**：在目标坐标上，加载适配器与干净基座的激活差分 L2 距离和余弦距离。
- **residual-L2 / residual-cos**：在最终层 residual stream 上计算的同类差分。
- **PPL-ratio**：`PPL_adapter(x) / PPL_base(x)`，即适配器与干净基座困惑度之比。
- **ONION（maxdrop）**：`max_i[PPL(x) - PPL(x_{-i})]`，其中 `x_{-i}` 为删除第 `i` 个 token 后的文本。
- **STRIP-p**：删除 20% 词汇并进行 8 次扰动时，最大类别概率的均值。

AUROC 以触发输入分数为正组、干净输入分数为负组，采用平均秩处理并列并计算 Mann–Whitney 统计量。所有分数均按预定方向解释为越大越异常。阈值仅由干净校准分数的第 95 百分位确定，测试集不参与阈值拟合。

## 5. 层级 paired bootstrap 与多重比较

主比较使用 `B=2000` 次层级 paired bootstrap：外层重抽三个训练种子，内层在被抽中的种子内重抽句对索引；两个适配器共享种子序列与句对索引。观察量先逐种子计算 AUROC，再等权平均。95% CI 为 bootstrap 差值的 0.025 与 0.975 分位数，即双侧 percentile 区间。

令 `kle` 与 `kge` 分别表示 bootstrap 差值不大于和不少于 0 的次数，零差值同时计入两尾。统计文件保留三种 p 值口径：

- `floor = max(2*min(kle/B, kge/B), 1/B)`
- `two_sided_add_one = min(1, (2*min(kle, kge) + 1)/(B+1))`
- `per_tail_add_one = min(1, 2*min((kle+1)/(B+1), (kge+1)/(B+1)))`

正文报告 floor-p 在预声明比较族内的 Benjamini–Hochberg 校正值 q。完整比较族标识、原始 p、q、尾部计数与可复现子种子保存在 `results/main/contrast_stats_complete.json`；相关敏感性口径保存在 `results/main/stats_sensitivity.json`。

## 6. 数值展示精度

正文中的 AUROC 差值及其 95% CI 原则上显示四位小数；接近零的差值和 q 值按需要保留更多有效位。此处仅为展示舍入，不改变原始结果。逐样本记录、逐种子结果和完整精度统计量以 `results/main/` 中的机器可读文件为准。
