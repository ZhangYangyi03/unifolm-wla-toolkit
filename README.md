# unifolm-wla-toolkit

数据侧工具：把自有机器人数据变成 UnifoLM-WLA-1.0 能直接吃的统一表示。
不需要 GPU，不需要真机，不需要模型权重。

依据的是官方规范 docs/robot_action_state_processing_en.md（unitreerobotics/unifolm-wla）。
宇树目前只放了 ER-1 / ER-Flow 权重，WLA-Base 权重和 post-train 代码都没放，
所以真正稀缺的不是"再训一个模型"，而是**把数据接进这套统一空间的管线**——
规范里写满了 must / must not，写错一位下标，模型学到的就是错的物理量。

## 文件

    unified_repr.py    规范实现：54 维动作 / 60 维状态、掩码、SE(3) 相对位姿、
                       旋转向量与 rotation-6D、归一化、抓夹二值化、重采样、
                       等任务权重统计量合并
    wla_prep.py        命令行：stats / convert / check / lerobot
    urdf_kin.py        零依赖 URDF 正运动学（公开数据集只有关节角，缺的就是这一步）
    unitree_g1_to_unified.py   G1 关节角 -> 统一布局的桥
    erflow_tokens.py   拿 ER-Flow 放出的 tokenizer 校验码本与标记
    erflow_prompt.py   从 payload 组一条 ER-Flow 输入序列（未公开处标为 unspecified）
    selftest.py / selftest_urdf.py / selftest_real.py   自检三件套
    run_all.py         一条命令 15 步全过
    demo/make_demo.py  造一份合成数据（桌面伸手 3 段），让管线立刻能跑
    demo/train_example.py  AutoDL 侧的读取示例（要 torch）
    docs/note_to_unitree.md  给宇树的说明：实现了什么、验证了什么、7 个待确认的空白

## 三分钟跑通

    python selftest.py
    python demo/make_demo.py demo/data
    python wla_prep.py stats   --src demo/data --out demo/data/stats.json
    python wla_prep.py convert --src demo/data --out demo/data/payload.npz --stats demo/data/stats.json
    python wla_prep.py check   --src demo/data --stats demo/data/stats.json

## 这套东西按规范实现了什么

动作是 54 维（H 步一块）：左右末端相对位姿各 6 维、抓夹各 1 维、灵巧手各 6 维、
腰 3、躯干 3、底盘速度 2、底盘偏航 1、底盘相对位姿 6、身高 1、左右腿各 6。
状态是 60 维：末端是绝对位姿 xyz + rotation-6D（取旋转矩阵前两列，行主序），
本体惯性槽是**重力方向 + 角速度**，不是绝对姿态。

容易写错的几条，这里都按规范做了：

- 末端/底盘的"相对动作"是 T_t^-1 @ T_t+k，平移在**当前末端坐标系**里，不是世界系差分。
- rotation-6D 一定取前两列，且约定为 [R00,R10,R20,R01,R11,R21]，重建用 Gram-Schmidt。
  混用列主序或后两列，训练能收敛但学出的姿态是错的。
- 相对位姿用 zscore（global_mean/global_std），普通量用 minmax_q，抓夹和灵巧手用
  gripper_norm_type，状态里的旋转 6 维**不归一化**（保持几何意义）。
- 尺度保护：|c| < 1e-6 时置 1，避免除零。
- 抓夹二值化在归一化之后：>0.9 记 1，<-0.9 记 0，中间值从序列尾往前填最近的确切状态。
- 处理顺序固定：相对位姿 → 归一化 → 抓夹二值化 → 重采样 → 填统一槽位。顺序换了两边训练不一致。
- 掩码：没启用的模块填 0 且 mask=0；输入维度不够**报错**，不自动补零（规范明确禁止）。
- 重采样越界取最近端点，不外推。
- 统计量是全局的（不分 chunk 内的位置），标准差用总体标准差（除以 N，不是 N-1）。

## 一条命令跑全部检查

    python run_all.py

15 步：单位几何恒等式、URDF 正运动学对拍（与 scipy 独立实现比）、合成数据全流程、
统计量、转换、审计、训练侧契约、真实 G1 端到端回环、ER-Flow tokenizer 校验、
ER-Flow 序列构建。不需要 GPU，不需要联网（真实数据和 ER-Flow 元数据都已在仓库里）。

## ER-Flow 那一半

WLA-Base 没放，但 **ER-Flow 的权重和 tokenizer 是放了的**，它吃的是离散动作码。
这部分我也接了，并且严格区分"已确定"和"未公开"：

    python erflow_tokens.py  --meta demo/er_flow_meta --stats demo/real_g1/src/stats.json
    python erflow_prompt.py  --meta demo/er_flow_meta --payload demo/real_g1/src/payload.npz \
        --out demo/real_g1/src/erflow_batch.json --horizon 2

已确定的（从放出的文件里读出来的，可校验）：tokenizer 里有 POS/ROT/HAND/LOW **四个
256 词表**，加 11 个控制标记（`<|EEF_START|>` 等），`robot_state_dim=120`，
`robot_state_projector: 120 -> 2560`。

**未公开的**（脚本不猜，只标出来）：54 维连续动作怎么映射到这些码、块内 token 顺序、
每个码本的值域、60 维状态怎么变成 120 维。`erflow_prompt.py` 用一个明确标注的
占位量化器（按本 payload 的 minmax 范围均匀分箱）跑通链路，输出的 JSON 里带
`unspecified` 字段。**别拿占位规则去训模型**——那是猜的。

## 给宇树科技的说明

`docs/note_to_unitree.md`：实现了什么、怎么验证的（15 项自检）、以及 7 个只能靠猜的
空白点（含英文摘要）。

已作为 issue 发给官方：https://github.com/unitreerobotics/unifolm-wla/issues/5
（同仓库 issue #2 是规范文档本身、#4 是 WLA-Base 权重时间线。）

仓库主页：https://github.com/ZhangYangyi03/unifolm-wla-toolkit

## 真实数据的用法（这条才是重点）

公开的宇树数据集（WLA-1.0 训练用的那批）是 LeRobot v3 格式，**只有关节角**，
没有末端位姿。而统一动作空间要的是末端/底盘位姿。中间缺的那一步是正运动学：

    python wla_prep.py lerobot --src <下载下来的数据集目录>      # 先看它到底有什么
    python unitree_g1_to_unified.py \
        --joints demo/real_g1/g1_joints_sample.npz \
        --urdf   demo/g1.urdf --out demo/real_g1/src --fps 30
    python wla_prep.py stats   --src demo/real_g1/src --out demo/real_g1/src/stats.json
    python wla_prep.py convert --src demo/real_g1/src --out demo/real_g1/src/payload.npz --stats demo/real_g1/src/stats.json
    python selftest_real.py

`wla_prep.py lerobot` 会直接告诉你那个数据集缺什么（例如"没有末端位姿，先做 FK"），
而不是把关节角硬塞进位姿槽位算出看着对其实是错的数。
`demo/g1.urdf` 是宇树官方 g1_29dof.urdf，`demo/real_g1/` 是 G1_Dex3_GraspSquare_Dataset
里真实取出的 300 帧。

## 源数据格式

    <src>/meta.json              {"fps":30, "pose_format":"rpy|quat|rotvec", "modules":[...]}
    <src>/episode_000000.npz
        cur/<module>     (T,d)      当前绝对观测；pose 模块是 xyz+rpy / xyz+quat(qx,qy,qz,qw) / xyz+rotvec
        action/<module>  (T,H,6|d)  未来 H 步目标，同一个坐标系下的绝对量

一个 ep 内 t 时刻的 cur 配 t+1..t+H 的动作；未来被截断的行直接丢掉，不补零。

## 产物

payload.npz：action (N,H,54) 已归一化、action_mask (N,54)、state (N,60)、state_mask (N,60)、
split、以及 action_offset/scale、state_offset/scale。
stats.json：按规范 7.2 的顶层格式（relative_action_key → global_max/min/q01/q99/mean/std）
外加 ordinary 统计与合并信息。offset/scale 是训练和部署**共用**的，务必一起存。

## AutoDL 上怎么用

数据侧在**本地跑**（这台的 CPU 就够，也不需要 CUDA），产物拷上去；GPU 只用来训练。

    # 本地
    python run_all.py
    python wla_prep.py convert --src <你的数据> --out payload.npz --stats stats.json
    # 传到 AutoDL
    scp payload.npz stats.json root@<autodl-host>:/root/wla/
    # AutoDL 上
    pip install -r requirements.txt
    python demo/train_example.py payload.npz

注意 AutoDL 上要装的是**训练**依赖（torch + transformers），数据侧只需要 numpy。
WLA-Base 的权重和 post-train 代码官方还没放；现在能在 AutoDL 上做的是
ER-1 / ER-Flow 的推理与微调（HF 上已有权重），或者用这份 payload 训自己的
小 policy 先把数据管线跑顺，等官方代码一放直接接上。

型号建议：ER-1-4B 微调 24G（4090/A5000）够；WLA-1.0 那个 6B 加 MMDiT 想全量微调
要 8 卡 A100/H100 级别，先用 LoRA 或者只训 action expert 更现实。

## 常见坑

- 归一化参数只存了归一化后的数据，部署时反归一化不一致 → 机器人动作幅度整体缩水。
- 每段任务单独算统计量再直接拼接 → 规范要求同本体跨任务按等任务权重合并后再用来归一化。
- 状态里的重力方向不做单位化检查 → |g| 不为 1，模型把姿态误差学成重力噪声。
- 左右末端统计量合并时没对齐坐标系 → 规范 sec. 13.3 要求先保证坐标系一致。

## Related work by the same author

The same claim -- *a number is meaningless until it is shown to survive its own
verification* -- is made and measured in other domains:

- [autoforge](https://github.com/ZhangYangyi03/autoforge) -- a tool's fitness, until an oracle outside the tool agrees
- [agentic-eda](https://github.com/ZhangYangyi03/agentic-eda) -- a circuit's area, until equivalence to the reference netlist is proven
- [debt-verify](https://github.com/ZhangYangyi03/debt-verify) -- a debt clause decision, until it survives the published revision record
- [tool-market](https://github.com/ZhangYangyi03/tool-market) -- a tool's liveness, until the hash chain says which revision is live
- [agent-safety-bench](https://github.com/ZhangYangyi03/agent-safety-bench) -- a model's safety compliance, measured rather than assumed
