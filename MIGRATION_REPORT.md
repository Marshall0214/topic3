# 第三点工程整理报告

日期：2026-10-13。方式：只读源目录、复制到 D:/topic3。未删除、移动或修改源文件。

## 复制范围

|源|目标|说明|
|---|---|---|
|D:/action_compare/keraal_localization/src、scripts、configs、tests、README|legacy/principal_v1|原样代码快照|
|同工程artifacts的清单、划分及审计|manifests/historical/principal_v1|保留原字段与时间假设|
|同工程artifacts/runs|runs/historical/principal_v1|包括约2.13GiB完整首轮权重、预测、缓存|
|同工程reports|reports/historical/principal_v1|原报告不改写|
|D:/AQA/work/keraal_dest 根文件|legacy/dest|所有原脚本、协议与诊断|
|同目录runs、pretrained|runs/historical/dest、upstream/dest_pretrained|历史实验和预训练文件|
|D:/AQA/research/keraal_literature/audit、根部文献|legacy/literature_20261011|小试代码、依赖、结果、论文及笔记|
|同目录upstream|upstream|作者源码，保留现有完整性，不伪称全部仓库齐全|
|D:/AQA/KeraalDataset|upstream/keraal_gmm|官方代码与GMM MATLAB示例；不复制.git|

共复制 **2031** 个文件，**2,693,073,516** 字节（约 **2.51 GiB**）。跳过.git、Python编译/JIT缓存与工具缓存目录、符号链接；不复制原始数据或conda环境。

## 工作副本与适配

- src/keraal_loc从原公共实现复制，算法字节保持一致。
- src/topic3提供data/features/alignment/reference/decoding/evaluation接口，当前复用旧Arms/Trunk单事件实现，不宣称四部位新模型。
- DeST工作副本只改路径入口和历史测试使用说明，模型、损失、选模及原阶段门槛保持。
- 单元检查只重定向清单位置；原断言保留。
- scripts/train.py显式要求新run名称，拒绝写入historical及路径穿越，默认annotation15。
- 原样快照与工作副本的差异见 [DERIVED_FILES.json](evidence/migration/DERIVED_FILES.json) 和 patches。

## 验证

- 全部2031份复制文件逐一SHA-256验证；源文件内容与修改时间在复制后重新核对，无变化。
- 7项既有检查通过：注释、几何归一化/旋转、DTW时间网格、事件与误报、解码秒数、划分/参考隔离、padding不变性。
- DeST合成数据前后向自检通过；无优化器训练。
- 保存的clock15结果重新汇总计数一致；未重推理测试集。
- 最后再次执行源与原样快照校验，2031份全部通过。
- 检查日志：[workspace_checks.log](reports/workspace_checks.log)。完整映射：[COPY_MANIFEST.json](evidence/migration/COPY_MANIFEST.json)。

## 本次发现并更正的台账遗漏

原action_compare工程已有完整首轮神经参考对齐、几何DTW和欧氏GMM实验；原文献audit也已有A-MAL约束DTW及线性控制。本次全部收录。作者流形GMM未完整运行与欧氏适配已运行是两回事；旧约束对齐已尝试与修正时钟的新统一基线未完成也应分开。

目录统一已完成。实验协议统一、作者流形GMM完整执行、修正时钟后的几何对齐公平比较仍属下一阶段。源目录不会因为迁移自动停用，但后续工作入口应使用topic3，原样legacy脚本不要直接运行。
