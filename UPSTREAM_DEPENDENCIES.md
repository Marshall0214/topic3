# Git仓库内容与本地依赖

本仓库保存项目代码、实验配置、研究计划和小型索引。原始数据、权重、完整预测、缓存、论文PDF、第三方源码副本和迁移归档留在本机，不随Git推送。忽略文件仍在本地，未删除或移动。

## 固定的作者源码

版本记录在configs/upstream_sources.json。当前工作区已有副本，不需要再次下载。新克隆仓库时，只有目标目录尚不存在才执行下面命令；已有目录不要覆盖或重置。

在项目根目录的新克隆中，逐行执行：

```powershell
git clone https://github.com/lyhisme/DeST.git upstream/dest
git -C upstream/dest checkout --detach 319e1f10f310e08a198fba3a32a4d82416ecb15f
```

可选ExeChecker审计：

```powershell
git clone https://github.com/EvenGu/ExeChecker.git upstream/ExeChecker_snapshot
git -C upstream/ExeChecker_snapshot checkout --detach 1e6df144bb49fd1173b027115d07a332f018c682
```

作者源码以各自许可证为准。本仓库不声称上述作者模型原生解决Keraal错误时空定位；具体适配与缺口见MODERN_BASELINE_AUDIT.md。

## 数据与历史文件

原始Keraal通过configs/paths.json指向外部目录，默认D:/AQA/dataset/keraal，保持只读。必要的匿名历史划分ID保留在manifests/historical/principal_v1/splits.json；含原始路径/逐记录标注的manifest.jsonl和audit.json不上传。

全新克隆不包含本机manifest、历史权重、reports/historical、evidence/migration或legacy，因此依赖这些文件的历史训练、历史汇总、迁移校验与部分契约检查不能直接执行。先恢复本机归档或完成plan.md的S0新协议生成；不能把缺少本地资产解释成已完成新训练入口。

scripts/check_environment.py可只读盘点依赖。scripts/check_modern_baseline_interfaces.py dest需要先恢复DeST源码及相应Python依赖。scripts/run_checks.ps1是本机归档验收流程，包含历史数据和迁移检查，不是无数据的新克隆自检。

scripts/organize_snapshot.py和setup_workspace.py为一次性整理追溯工具，不在新克隆中盲目执行；它们可能依赖旧绝对路径。原始数据、作者源码和研究代码的许可分别处理。

## 新实验的版本管理

提交src、scripts、tests以及经过审查的配置和总结。大型runs和逐帧预测只保存在本机。每次实验在小型配置/报告中记录代码版本、数据划分、参数、种子与输出位置；后续需要发布精选结果时单独审查，不全量取消忽略规则。
