# 给宇树科技的一封说明 / A note to Unitree Robotics

发送对象：unitreerobotics/unifolm-wla issue 区（或 UnifoLM 团队邮箱）
日期：2026-09-16
作者：ZhangYangyi03（独立研究者，非宇树合作方）
配套仓库：https://github.com/ZhangYangyi03/unifolm-wla-toolkit

---

## 先说清楚这是什么

我读了 `docs/robot_action_state_processing_en.md`，把里面的规范**逐条实现成了一版可运行的
数据侧工具**，并用宇树自己公开的 G1 数据跑通了端到端回环。目的不是提意见，是：

1. 证明这份规范是可以照着实现的、没有自相矛盾（如果它能被一个人实现出来并被 15 项
   独立检查通过，它就是我们这些外部使用者能用的文档）；
2. 把实现过程中**只能靠猜**的地方一条条列出来，请你们确认——这些地方猜错的代价不是
   报错，而是训练照常收敛、学出来的物理量是错的。

仓库里所有结果都可以在**没有 GPU、没有真机、没有联网**的机器上复现：

    git clone https://github.com/ZhangYangyi03/unifolm-wla-toolkit
    python run_all.py      # 15/15

## 我实现了什么

- 54 维动作 / 60 维状态、掩码语义、SE(3) 相对位姿（T_t^-1 T_{t+k}，平移在当前末端系）、
  rotation-6D（前两列、行主序、Gram-Schmidt 重建）、zscore / minmax_q / gripper 归一化、
  抓夹二值化的阈值与补位规则、重采样端点保持、全局统计量与**等任务权重**合并。
- URDF 正运动学（零依赖），因为公开的 G1 数据集是 LeRobot v3 格式、**只有关节角**，
  而统一空间要位姿——中间这一步没有人补，我自己补上了，并用 scipy 独立实现对拍到 4.4e-16。
- ER-Flow 的离散动作侧：你们放出的 tokenizer 里有四个 256 词表的码本
  （POS / ROT / HAND / LOW）和 11 个控制标记，我也读出来并做了校验。

## 我验证了什么（不需要你们相信我的话）

- 相对位姿恒等式：从归一化后的 payload 反解回绝对位姿，全 300 帧最大偏差 2.4e-8。
- 统计量合并：两个任务（10 个样本 vs 1000 个样本）合并后均值为 5 而不是 9.9，确认是
  等任务权重；合并方差满足全方差公式（sqrt(26)=5.0990）。
- 真实数据：G1_Dex3_GraspSquare 的 300 帧走完 FK → 统一布局 → 统计 → 归一化 → 反解，
  末端速度 0.22 m/s（量的级次合理）。
- 失败也是特性：`wla_prep.py lerobot` 读到没有末端位姿的数据集会**返回 1 并说明原因**，
  而不是把关节角塞进位姿槽位算出"看着对"的数。这条我写进了自检。

## 请你们确认的 7 件事（按重要性排）

### 1. 公开数据集缺末端位姿，是设计如此还是待补？

`unitreerobotics/G1_Dex3_GraspSquare_Dataset` 这类 LeRobot v3 数据只有关节角。
规范 4.2 要求末端相对位姿。官方是否打算发布"已做 FK 并转好"的版本，还是使用者各自实现？
如果各自实现，那么**每个人的 URDF 版本 / 关节命名 / 工具系定义不同，统计量就不可比**——
这会直接影响规范 12.1 的跨任务合并。

### 2. 54 维连续动作 → POS/ROT/HAND/LOW 离散码，映射规则是什么？

ER-Flow 的 README 说三部分各训一个 RVQ，但**权重没放**，也没说 54 维的哪几维进哪个码本、
每个码本的值域是多少、token 在序列里的排列顺序（虽然有 `<|EEF_START|>` 这类标记，
块内布局未公开）。
我在 `erflow_prompt.py` 里用"按本 payload 的 minmax 范围均匀分箱"作为**占位**实现，
并在输出的 JSON 中把它标成 `unspecified`，以免有人误当成官方规则去训。请给出真实规则或权重。

### 3. 60 维统一状态 → `robot_state_dim=120`，另外 60 维是什么？

`config.json` 里 `robot_state_dim: 120`，`robot_state_projector: 120 → 2560`，
而规范的状态空间是 60 维。是双手机型×2、还是 60 state + 60 另一路（如本体惯性/历史），
还是没有归一化的原始量？这个数字对不上，任何按规范产出 60 维的人都会卡在这里。

### 4. `<|robot_state_implicit_stats|>` 是什么？

tokenizer 里有这个 token，规范里没提。是不是"用隐式统计量替代显式 offset/scale"？
如果是，训练和部署时的统计量从哪来？

### 5. 等任务权重合并，官方是否真的按任务均匀采样？

规范 12.1 明确"每个任务权重相同、样本数不影响权重"。但如果训练实际是**按帧均匀采样**，
等任务权重会让长任务被低估。我按规范实现了等任务权重，并提供 `weights=` 逃生口。
想知道 WLA-1.0 预训练时实际用的是哪种。

### 6. 状态里 rotation-6D 不归一化，和底座 inertial 槽位的关系

规范 11.1 说末端旋转 6 维不归一化（这我完全认同），`[41:47]` 是重力方向+归一化角速度
（不是姿态）。但 11 节的表格里 `[48:54]` 左腿用"左腿状态统计量的前 6 个分量"——
腿的"状态"具体是哪 6 个自由度、顺序如何？G1 是 29 自由度，腿 12 个，取哪 6 个？

### 7. 部署侧的 offset/scale 分发

规范要求 offset/scale 训练部署共用。官方会不会随 checkpoint 一起发一份
"本模型对应的 statistics.json"？不发的话，第三方微调后的模型换一组统计量就不可复现。

## 我想做的下一步（如果你们允许）

- 把 `unifolm-wla-toolkit` 的检查接进你们的 CI：任何规范改动，15 项自检立刻告诉你们
  哪条被破坏；
- 若放出 WLA-Base 权重与 post-train 代码，我可以把这条管线直接接到你们的训练入口上，
  并在 G1 数据上给出可复现的对比；
- 我这边有 AutoDL 云 GPU（A100 级别），可以承担 ER-1/ER-Flow 的复现实验，
  结果公开可查。

## 说明

这不是 PR，也不是要求。我没有你们的模型权重，也没有真机。我能给的是一份**可复现的、
把规范当合同来执行的实现**，以及实现过程中暴露出来的 7 个空白点。
如果这 7 条里有任何一条的答案已经在某处，请指个路，我改。

---

## English summary (for the repo)

I implemented `docs/robot_action_state_processing_en.md` as runnable code and validated it
end to end on Unitree's own public G1 data — no GPU, no robot, offline:

    python run_all.py      # 15/15 checks

Highlights: relative pose identity recovers absolute poses to 2.4e-8 over 300 real frames;
statistics merging is equal-task-weight (10-sample task vs 1000-sample task merges to mean 5,
not 9.9); URDF forward kinematics cross-checked against an independent scipy implementation to
4.4e-16; the LeRobot reader refuses (exit 1, with a reason) rather than forcing joint angles
into pose slots. I also read the released ER-Flow tokenizer: four 256-entry codebooks
(POS/ROT/HAND/LOW) and 11 control markers, verified present and contiguous.

Seven questions where the published material does not determine the answer, and where guessing
wrong is silent rather than loud:

1. Public G1 datasets ship joint angles only; the unified space needs end-effector poses. Will
   official FK-converted datasets be released, or is each consumer expected to roll their own?
   Different URDFs/frames make statistics non-comparable across users, which bears directly on
   the sec. 12.1 merge rule.
2. The 54-dim continuous action -> POS/ROT/HAND/LOW discrete codes mapping (RVQ weights not
   released, per-codebook ranges and intra-block token order unpublished).
3. `robot_state_dim = 120` vs the spec's 60-dim state: what are the other 60?
4. What is `<|robot_state_implicit_stats|>` for?
5. Is training actually task-balanced, or uniformly sampled over frames? (spec 12.1 assumes the
   former)
6. Which 6 leg components are `[48:54]` for a 29-DoF G1, and in what order?
7. Will the offset/scale statistics be distributed with checkpoints?

`erflow_prompt.py` builds an ER-Flow input from a real payload, uses a clearly-labelled
placeholder quantizer where the real rule is unpublished, and lists those gaps in its own
output so nobody trains on a guess.
