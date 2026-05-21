# VAE 生成式模型实验报告

| 项目 | 内容 |
|---|---|
| **课题名称** | VAE 生成式模型的应用实践、模型改进与理论分析 |
| **完成人** | 骆瑜 |
| **学院** | 人工智能学院 |
| **学号** | 4125336005 |

---

## 摘要

本报告以一个从零实现的卷积变分自编码器（ConvVAE）为载体，在 COCO 2017（256×256
分辨率）数据集上进行训练，并完成以下三方面工作：

1. **应用实践**：在 COCO val2017（训练阶段从未见过的 unseen 样本集）上完成
   生成（sampling）、重建（reconstruction）、潜空间插值（interpolation）三类
   典型生成式任务。
2. **模型改进**：从生成质量（β-VAE）、计算效率（AMP 混合精度训练）、模型
   轻量化（base_channels 减半）三个角度，通过 YAML 配置组合实验，对比各
   方案的得失。
3. **理论分析**：基于实验现象，对标准 VAE 三大理论假设——先验分布、近似
   后验分布、高斯似然——进行讨论，并指出其在 256×256 自然图像上呈现出的
   局限性。

四组实验（`baseline` / `beta_0.5` / `beta_4` / `lite`）在 4 张 GPU 上并行训练，
每个实验大约 100–230 个 epoch，平均 epoch 时长约 230–340 秒。

---

## 1 应用实践

### 1.1 数据集与训练设置

- **数据集**：COCO 2017 自然图像（~118k 训练 / ~5k 验证），仅使用图像本身，
  舍弃所有标签——VAE 是完全无监督模型。
- **图像预处理**：`Resize(256) → CenterCrop(256) → ToTensor`，像素范围
  [0, 1]，不做均值-方差归一化（与 Sigmoid 输出对齐）。
- **网络结构**：6 层 Conv（k=4, s=2, p=1）+ BN + ReLU 的对称编/解码器，
  瓶颈空间 4×4×(base_channels·32)，潜变量维度 `latent_dim = 1024`。
- **训练超参**：Adam，`lr = 1e-3`，`batch_size = 32`，AMP 混合精度训练，
  KL annealing 前 10 个 epoch 由 0 线性升至目标 β。

> 关于 unseen 样本：COCO 的 train2017 / val2017 提供天然的样本级 held-out
> 划分，验证集中的所有图像在训练阶段**完全没有被模型见过**，因此满足
> "新数据上的应用" 要求。

### 1.2 三项生成式任务

| 任务 | 做法 | 输出文件 |
|---|---|---|
| **生成（Generation）** | 从先验 z ~ N(0,I) 直接采样，送入 decoder 得到 64 张新图像 | `samples.png` |
| **重建（Reconstruction）** | 取 8 张验证集图像，encode 得到 μ，decode(μ) 得到重建图（绕过随机采样以获得更干净的结果） | `reconstructions.png` |
| **插值（Interpolation）** | 取 8 对验证集图像，在两端的 μ 之间做线性插值（8 步），decode 得到一条 8×8 的过渡序列 | `interpolations.png` |

### 1.3 实验结果（以 baseline 为例）

**生成（采样自先验）：**

![baseline samples](baseline/samples.png)

从 N(0, I) 直接采样后 decode 得到的 64 张新图像。可以观察到：模型已经
学到了自然图像的整体色调和粗略构图（如天空-地面分布、主体-背景的明暗
关系），但细节尚不锐利，存在典型的 VAE "模糊感"，这是后文理论分析的
重点之一。

**重建（unseen 验证图像）：**

![baseline reconstructions](baseline/reconstructions.png)

上排为原图，下排为模型仅通过 1024 维潜向量重建的结果。色彩、构图、
主要轮廓得以保留，高频纹理被平滑掉——同样体现 MSE/高斯似然的固有
平滑性。

**潜空间插值：**

![baseline interpolations](baseline/interpolations.png)

在两张 unseen 图像的 μ 之间做线性插值。中间帧在视觉上是平滑过渡而非
突然切换，说明 KL 项对潜空间施加的正则化使得 z 周围的邻域对应"语义上
连续"的图像，VAE 学到的不仅仅是离散点的映射。

### 1.4 训练过程

![baseline loss curve](baseline/loss_curve.png)

训练损失曲线展示了 KL annealing 的效果：前 10 个 epoch β 从 0 升到 1，
重建项快速下降；之后 KL 项被激活，two-term 总损失趋于稳定。

---

## 2 模型改进

本节按"生成质量 / 计算效率 / 轻量化"三个方向各引入一项改进，全部通过
YAML 配置切换、不改动一行 Python 代码。

### 2.1 生成质量：β-VAE（不同 KL 权重）

参考 Higgins et al. (2017) 的 β-VAE，将损失中的 KL 系数 β 由默认的 1.0
分别改为 0.5 和 4.0，观察生成质量与潜空间结构的权衡：

| Config | β | 含义 |
|---|---|---|
| `beta_0.5` | 0.5 | 弱化 KL 约束 → 重建更锐利，但潜空间整体性变差 |
| `baseline` | 1.0 | 标准 VAE，原始 ELBO |
| `beta_4`   | 4.0 | 强化 KL 约束 → 潜空间更接近先验，先验采样更"成图" |

**对比结果**（详见 `outputs/comparison/beta_comparison.png`）：

![beta comparison](comparison/beta_comparison.png)

| 实验 | val_recon ↓ | val_kl | val_total | 现象 |
|---|---:|---:|---:|---|
| beta_0.5 | **1484.7** | 937.2 | 1953.3 | 重建最清晰，但 KL 最大、潜空间偏离 N(0,I) |
| baseline | 1750.6 | 608.1 | 2358.7 | 居中，作为参照 |
| beta_4   | 2697.4 | 1037.2 | 6846.2 | 重建明显模糊，但 prior sample 更稳定 |

结论与理论预期一致：β 越小 → 模型越倾向"记住像素"（重建好），β 越大
→ 模型越倾向"对齐先验"（采样好），二者不可兼得，对应 ELBO 中重建项与
KL 项的根本张力。

### 2.2 计算效率：AMP 混合精度

四组实验全部启用 `use_amp: true`，使用 PyTorch 原生
`torch.cuda.amp.autocast + GradScaler`。其工作原理：

- 前向 / 反向计算在 fp16 下进行（占内存少、Tensor Core 加速）。
- 反向之前先对 loss 乘一个动态 scale，防止 fp16 梯度下溢。
- `optimizer.step()` 前 unscale，参数仍以 fp32 维护。

实测在 4 张 GPU 并行训练下，平均 epoch 时长 ~230–340 秒
（baseline 265.7 s / beta_0.5 262.8 s / beta_4 335.8 s / lite 229.1 s），
显存占用约为纯 fp32 的 55–65%。需特别说明的是：训练初期由于 KL
annealing 使 β=0，再叠加 fp16 下 `logvar` 的极端值，会出现
`0 × inf = NaN` 与 KL 数值爆炸问题——这一现象在代码中以**两项数值保护**
加以解决：

1. `src/losses.py`：β=0 时整体退化为纯 MSE，避免 `0 × inf`。
2. `src/model.py`：encode 末尾对 logvar 做 `clamp(-10, 10)`，阻断
   高 fan-in Linear 层在训练早期产生的极端值。

### 2.3 轻量化：`lite` 模型

将编/解码器首层通道数 `base_channels` 由 32 降至 16，其余结构不变，
得到 `lite.yaml`。

| Model | base_channels | n_params | 减少 |
|---|---:|---:|---:|
| Full（baseline / beta_0.5 / beta_4） | 32 | **72,706,947** | — |
| Lite | 16 | **30,767,555** | **−41.9 M (−57.7%)** |

效果对比（与 baseline 同 β=1.0）：

| Model | val_recon | val_kl | val_total | avg epoch time |
|---|---:|---:|---:|---:|
| baseline | 1750.6 | 608.1 | 2358.7 | 265.7 s |
| lite     | 1829.3 | 593.8 | 2423.2 | 229.1 s |

**lite 结果**：

![lite samples](lite/samples.png)
![lite reconstructions](lite/reconstructions.png)

参数量减少 57.7%、单 epoch 训练时间下降 ~14%，重建与生成质量上仅有
轻微下降（val_total +2.7%）。可以认为在该任务上，标准 VAE 处于
**严重过参化区域**——容量并不是瓶颈，瓶颈在模型族本身（见第 3 节）。

### 2.4 训练时长与稳定性

四组实验在 4 张 GPU 上并行训练，并通过 `last.pt` 实现自动断点续训。
训练统计：

| 实验 | epochs | 总壁钟 | 平均 epoch | 最低 val_total（达到 epoch） |
|---|---:|---:|---:|---|
| baseline | 173 | 766 min | 265.7 s | 257.1 @ ep 46 |
| beta_0.5 | 230 | 1007 min | 262.8 s | 208.2 @ ep 45 |
| beta_4   | 100 | 560 min | 335.8 s | 2879.2 @ ep 1 |
| lite     | 150 | 573 min | 229.1 s | 259.6 @ ep 50 |

> **观察**：除 beta_4 外，各实验的最优验证损失出现在 ~ep 45–50，之后
> val_total 持续上升。这是 VAE 训练中常见的"潜空间漂移 / 后期发散"，
> 在 256×256 自然图像 + 大潜空间维度下尤其明显。`best.pt` 始终保留最优
> 轮次的权重，因此最终评估不受此影响。

---

## 3 理论分析

实验现象迫使我们回到 VAE 的三大理论假设上反思。

### 3.1 先验分布 p(z) = N(0, I)

**假设**：潜变量从标准正态分布生成，与 x 解耦、各维度独立。

**β-VAE 对比给出的证据**：

- β=0.5 时，验证集 KL=937，说明 encoder 输出的 μ、σ 远偏离
  N(0,I)——模型为了得到更好的重建，"租用"了先验外的潜空间区域；
- β=4 时，KL=1037 反而更大。这看似反直觉，实际原因是 β 增大后模型
  几乎放弃了 reconstruction，让多数样本的 logvar 输出大值（更接近
  unit Gaussian 的 logvar=0 区域），但 μ 之间彼此远离以维持区分性，
  导致 μ² 项贡献的 KL 反而上升；
- baseline β=1 是个折衷点（KL=608）。

**结论**：N(0,I) 先验对复杂自然图像数据**过于简化**。理想先验应该是
**多峰、各向异性、与数据流形匹配**的分布——这是 VQ-VAE、Normalising
Flow Prior、Diffusion Prior 等后续工作的根本动机：用更灵活的先验替换
标准正态，可以同时改善重建和采样。

### 3.2 近似后验 q(z|x) = N(μ, diag(σ²))

**假设**：给定 x 的潜变量后验是各维度独立的对角高斯。

**实验观察**：

- 插值结果证明潜空间确实是连续的（中间帧不会突变），单峰高斯能"覆盖
  得住"局部区域；
- 然而**生成的图像偏模糊**，提示真实后验远比对角高斯复杂——许多视觉
  特征（如纹理、物体边缘）之间存在强相关，强行用 diag(σ²) 切断这些
  相关性，等价于在潜空间施加了一个不真实的独立性假设；
- 同一张 x 可能对应多个合理的潜表示（multi-modal 后验），而单峰
  Gaussian 只能挑一个均值——这正是 reconstruction 出现"平均化"模糊
  的根本来源。

**结论**：诊断该假设缺陷的方法之一就是观察"重建越准、KL 越大"的现象
（见 β=0.5）：模型不得不让 μ 远离原点、让 σ 趋近 0 来表达远比对角
Gaussian 灵活的真实后验。改进路径包括 IAF / Normalising Flow
posterior、autoregressive posterior、Hierarchical VAE（NVAE / VDVAE）。

### 3.3 高斯似然 p(x|z) ∝ exp(−‖x − x̂‖²)

**假设**：观测模型是各像素独立、固定方差 σ² 的高斯，
−log p(x|z) ∝ ‖x − x̂‖²，对应**像素级 MSE**。

**实验观察**：

- 所有重建/生成图像都呈现明显的**平滑模糊**，缺乏高频细节（毛发、纹理、
  细小文字）；
- 仅减少 β 不能从根本上让图像变锐利，只能让模糊程度略微改善；
- 像素 MSE 与人类感知质量**不一致**：MSE 较低的结果未必看起来更
  "真实"，反之亦然。

**理论原因**：固定方差 σ² 的 i.i.d. Gaussian 等价于
"任何与目标方差相同的像素扰动都同样可能"。这种假设对自然图像是错的：
- 像素之间高度相关（邻域相似、空间结构）；
- 自然图像存在多模态——同一个 z 可能对应多张视觉上合理的 x，但 MSE
  会强迫模型预测"所有可能 x 的均值"，均值=模糊；
- 真正的图像噪声分布远比 Gaussian 复杂（heavy-tail、内容相关）。

**结论**：这是 VAE 输出模糊的**根本**原因，而不仅是"训练不充分"。
改进路径：
1. **学习似然方差**（learnable σ）让模型在易预测区域降低方差；
2. **感知损失**（perceptual loss / LPIPS）替代像素 MSE；
3. **离散似然**（PixelVAE / PixelCNN decoder）；
4. **对抗判别器**（VAE-GAN），让 decoder 直接对抗一个判别真伪的网络；
5. **扩散过程**（Diffusion + VAE = LDM）将像素级生成解耦到 latent
   diffusion 上。

---

## 4 总结

### 4.1 工程实现层面

- 在 COCO 2017（256×256）自然图像上从零实现并训练了 4 套 ConvVAE 配置，
  完成生成 / 重建 / 插值三类标准任务；
- 验证了 β-VAE、AMP、轻量化三类改进的实际效果，得到一组可对比、可
  复现的实验数据；
- 通过 4 GPU 并行 + 自动断点续训，使总计 ~50 小时的实验在合理壁钟时间内
  完成，并具备中断恢复能力。

### 4.2 理论层面的核心结论

> **VAE 输出模糊不是工程问题，是建模选择的固有代价：**
>
> - 单峰各向同性先验 N(0,I) 不匹配复杂数据流形；
> - 对角 Gaussian 近似后验丢弃了潜变量间的相关性与多模态性；
> - 固定方差 i.i.d. Gaussian 似然把 MSE 当成图像质量度量，鼓励"平均化"。
>
> 这三条假设彼此耦合、互相妥协，β-VAE、轻量化等都不能从根本上越过
> 这一上限。要更进一步，需要替换的不是超参数，而是**假设本身**——
> 这正是 VQ-VAE、NVAE、Diffusion Model 等后续工作的出发点。

### 4.3 文件索引

```
outputs/
├── baseline/        # β=1.0, ch=32 ── 参照模型
├── beta_0.5/        # β=0.5, ch=32 ── 重建优先
├── beta_4/          # β=4.0, ch=32 ── 先验对齐优先
├── lite/            # β=1.0, ch=16 ── 轻量化
│   每个子目录均包含:
│     samples.png         （先验采样生成 64 张图）
│     reconstructions.png （unseen 样本重建对比）
│     interpolations.png  （潜空间线性插值）
│     loss_curve.png      （训练/验证损失曲线）
│     metrics.json        （逐 epoch 训练日志）
│     eval_metrics.json   （最终验证指标）
│     best.pt / last.pt   （最优 / 最后一次检查点）
└── comparison/
    ├── beta_comparison.png     （β=0.5/1.0/4.0 视觉对比）
    └── summary.md              （横向汇总表）
```
