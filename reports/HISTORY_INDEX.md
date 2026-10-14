# 历史实验索引

所有路径相对于 D:/topic3。历史输出保留原内容。本次未重新推理、重训或调阈值。

|实验|代码/协议|报告|权重、预测与日志|评价状态|
|---|---|---|---|---|
|完整首轮：无参考TCN、参考TCN、几何DTW、欧氏GMM|legacy/principal_v1；manifests/historical/principal_v1；configs/legacy_principal_v1|[完整报告](historical/principal_v1/EXPERIMENT_REPORT.md)|runs/historical/principal_v1/principal_cpu_v1|旧10fps映射；旧测试已查看，不能视为新时钟可靠基准|
|完整首轮GPU试跑|同上|同上说明|runs/historical/principal_v1/principal_v1|原GPU探索，不混入CPU主结果|
|RF-DTAN＋线性/DTW控制|legacy/literature_20261011/audit/alignment_pilot.py|[报告](../legacy/literature_20261011/audit/ALIGNMENT_PILOT_REPORT.md)|同audit下alignment_pilot_ctk、alignment_pilot_ctk_e200|旧时钟、CTK单折|
|A-MAL兴趣点适配|legacy/literature_20261011/audit/amal_pilot.py|[报告](../legacy/literature_20261011/audit/AMAL_PILOT_REPORT.md)|同audit下amal_pilot_ctk*|旧时钟、训练子集|
|A-MAL约束DTW/局部解码/线性控制|同audit下amal_dtw_optimized.py等|[报告](../legacy/literature_20261011/audit/AMAL_DTW_OPTIMIZATION_REPORT.md)|同audit下amal_dtw_optimized_ctk、amal_linear_control_ctk|旧时钟；校准全拒绝也保留|
|DeST修正时钟时间定位|legacy/dest/stage1.py；当前适配src/topic3/models/dest_stage1.py|[时钟实验报告](../legacy/dest/CLOCK15_RESULTS_20261011.md)|runs/historical/dest/stage1_clock15|开发验证、折间重叠、非独立测试|
|DeST局部帧头|同上|原报告状态过时，以fold结果为准|runs/historical/dest/stage1_clock15_local|三折已完成，仅时间定位|
|其余DeST GN/参考/预训练/平衡损失|legacy/dest|对应原协议与runs中的summary|runs/historical/dest其他子目录|encoded旧时钟，保留作探索|
|ExeChecker/DyFADet兼容检查|legacy/literature_20261011/audit|原审计JSON和文档|同目录；作者源码在upstream|接口兼容，不是有效性验证|

原权重使用其原始结构和特征；不能因为位置迁移就用新特征直接加载比较。历史报告内链接可能仍指向原目录，本文提供新位置导航。
