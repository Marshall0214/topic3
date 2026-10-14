# 实际使用 Keraal 的工作：数据增强证据核查

核查日期：2026-10-14。用户要求先参考实际使用 Keraal 的工作，本轮暂停增强实现。没有新训练，没有验证/测试集推理，没有修改源数据或作者代码。

## 核查结论

本轮核实的Keraal工作没有提供可直接照搬到错误时间区间＋身体部位联合定位的、经过消融验证的增强配方。这个结论限于下列核查材料，不代表领域不存在相关增强研究，也不证明增强无效。

|工作|Keraal上的任务|增强证据|对本任务的意义|
|---|---|---|---|
|PhysioFormer：A Spatio-Temporal Transformer for Physical Rehabilitation Assessment，ICSR+AI 2024，论文集2025|骨架动作质量评估，Keraal正误评估报告F1|作者全文§3.3，论文内页码4—5跨页，明确从SkateFormer流程移除了data augmentation|原动作识别模型有增强，不代表康复适配沿用了它；未核实单独的增强消融或删除原因|
|HyperPhysio：Skeleton-Based Transformer for Classification of Errors and Better Feedback in Low Back Pain Physical Rehabilitation Exercises，2025|每个动作单独训练，正确＋三种错误分类|§III-B把序列插值到统一长度；§VI将探索增强列为未来方向|统一长度是预处理，不是已验证的随机增强；不能把原Hyperformer增强参数归给该论文|
|DeepRehabPile：A Standardized Benchmark for Skeleton-Based Rehabilitation Assessment Using Deep Learning，arXiv 2507.21018当前公开版|Keraal三动作的二分类与多分类，共6个分类子任务|§5.1报告Fourier统一长度、仅训练集拟合min-max、跨人员划分；未报告Keraal专门的增强收益或联合定位增强消融|可以借鉴数据协议；不能将分类档案当成带逐帧错误标注的数据，也不能将模型集成当作数据增强|
|Analyzing Data Efficiency and Performance of Machine Learning Algorithms for Assessing Low Back Pain Physical Rehabilitation Exercises，ECMR2023／预印本2024|Keraal正确性评估，比较Kinect/OpenPose/BlazePose及GMM/STGCN|全文未报告明确的随机增强配方与其消融|不同传感器/姿态估计器是同一执行的不同观测，不能作为互相独立的受试者样本|

证据层级：PhysioFormer的移除是明确陈述；HyperPhysio将增强列未来是明确陈述；后两项是未报告/未核实，不等于证明作者所有内部代码绝对无增强。

## 两条准确的原文证据

- PhysioFormer §3.3："We adapt their model to our regression problem, and remove data augmentation from the pipeline"。作者公开PDF正文内印刷页码4页末接5页首；不是根据SkateFormer原仓库推断。
- HyperPhysio §VI："Exploring data augmentation methods could lead to better classification"。上下文将此列为后续研究。

## 必须排除的混淆

《Weighted Average of Human Motion Sequences for Improving Rehabilitation Assessment》介绍中提到Keraal，但§4.1的增强实验实际使用KIMORE，不能列为“已在Keraal验证的增强方法”。其加权运动平均思想只能放在外部候选来源。

Keraal Group3是健康人真实执行和模拟预定义错误的实采记录，不是算法生成的数据增强。HyperPhysio Scenario1以Group3训练并在Group1A/2A评价，这是训练数据组织和跨域评价，不是时空标签保持的增强证据。

本地Group3有整段正确性/错误类型信息，没有已经核实的逐条错误起止标注。若将其作为额外训练数据，可探索分类预训练后迁移编码器，但不能给整段随意填上错误区间，也不能自动参加本任务的事件监督。须避免人员/原执行跨划分泄漏，另作“额外数据/预训练”消融。

## 对现有方案的修正

DATA_AUGMENTATION_PLAN.md中的局部时间变形、微扰、参考多样化仍为本项目候选，尚无本轮Keraal直接文献验证，不写成作者推荐配方或已成立创新。

保留A0无增强基线；先核查实际错误标签语义与边界映射，再决定是否实施A1。不能根据“样本少”就假定任意增强有效；尤其改变手臂高度、躯干倾斜、旋转幅度会改变这些错误本身。普通分类标签保持也不足以证明错误部位与时间标签保持。

现阶段不启动GAN/扩散，不调用ExeChecker、DeST或GYMetricPose的外部增强来冒充Keraal已有证据。这些模型尚未在原论文中完成本任务的Keraal联合定位增强验证。

## 一手来源

- PhysioFormer作者公开全文：https://hal.science/hal-04857956/document
- PhysioFormer出版记录：https://doi.org/10.1007/978-981-96-3525-2_14
- HyperPhysio：https://arxiv.org/html/2504.13866v1
- DeepRehabPile：https://arxiv.org/html/2507.21018
- ECMR数据效率论文：https://arxiv.org/html/2408.02855v1
- Keraal数据集原文：https://arxiv.org/html/2407.00521
- 排除的KIMORE增强论文：https://maxime-devanne.com/publis/ismail-fawaz_aaltd2024.pdf

注：PhysioFormer正文已通过作者公开PDF实际读取；未因HAL浏览器工具暂时返回反爬页而只依据标题判断。出版页为付费预览，未购买或访问受限内容。
