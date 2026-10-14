# 第三研究点实验工程

总题目：《基于边缘计算的动作捕捉和分析方法研究》。第三点暂定：基于参考动作对齐的三维人体动作时空偏差定位。整理日期：2026-10-13。

**工程资料已集中到 D:/topic3；原目录保留，原始数据仍从 D:/AQA/dataset/keraal 读取。** 本次复制2031个文件，约2.51 GiB，源文件内容与修改时间校验通过。目录整理完成不等于新基线训练或时间协议统一已完成。

## 从这里阅读

- [UPSTREAM_DEPENDENCIES.md](UPSTREAM_DEPENDENCIES.md)：Git提交范围、作者源码固定版本和全新克隆所需的本地资产。

- **[plan.md](plan.md)**：当前完整实施主方案；统一任务、现代基线、Group3与增强消融、阶段验收及工程管理。其他文件保留细节与历史证据。

- [KERAAL_AUGMENTATION_RELATED_WORK.md](KERAAL_AUGMENTATION_RELATED_WORK.md)：实际使用Keraal的增强证据、明确未采用与未报告的区别；当前暂停实现。

- [DATA_AUGMENTATION_PLAN.md](DATA_AUGMENTATION_PLAN.md)：第三点标签保持的数据增强候选、时间映射约束及公平消融；仅设计，尚未启用。

- [MODERN_BASELINE_AUDIT.md](MODERN_BASELINE_AUDIT.md)：现代候选审计、完整DeST与旧适配差异、实测证据及主基线选择。

- [BASELINE_STRATEGY_CURRENT.md](BASELINE_STRATEGY_CURRENT.md)：当前主线，先选现代方法基线，再针对性优化；DTW为传统对照。

- [NEXT_STEP_PLAN.md](NEXT_STEP_PLAN.md)：传统几何对照的实施细节，优先级已按当前主线调整。

- [BASELINE_REGISTER.md](BASELINE_REGISTER.md)：基线状态、已有结果、整理时补充发现的完整首轮实验。
- [FEASIBILITY_REPORT.md](FEASIBILITY_REPORT.md)：可行性、任务边界与实验限制。
- [MIGRATION_REPORT.md](MIGRATION_REPORT.md)：源→目标对应、验证结果与待完善项。
- [历史实验索引](reports/HISTORY_INDEX.md)：已完成实验、权重和预测的位置。
- [运行入口说明](scripts/README.md)：哪些可用、哪些仍为历史归档。

## 目录

```text
topic3/
  configs/                     # 当前路径、环境快照；旧协议单独保存
  src/keraal_loc/               # 原公共实现的工作副本，算法未改
  src/topic3/                   # 统一接口和路径适配后的DeST
  scripts/                     # 环境、自检、训练入口、结果汇总、复制校验
  tests/                       # 迁移后的7项既有关键检查
  manifests/historical/        # 原清单、划分和审计，保留原字段
  upstream/                    # 作者源码、GMM示例、预训练权重
  runs/historical/             # 已有DeST与完整首轮权重/预测/日志
  runs/<新实验名>/             # 后续新结果；当前未启动
  reports/historical/          # 原完整首轮报告
  legacy/                      # 原始代码、旧文献审计及小试结果
  evidence/                    # 数据核查、来源、复制清单和补丁
```

## 一行检查命令

在任意 PowerShell 目录执行：

```powershell
& 'D:\topic3\scripts\run_checks.ps1'
```

它使用现有 `C:/Users/admin/anaconda3/python.exe`，执行环境盘点、7项既有检查、DeST合成前后向、保存结果汇总及2031份源/副本哈希核验。只读一条已有训练注释及清单用于契约检查；不训练、不做新测试集推理。

查看训练入口参数：

```powershell
& 'C:\Users\admin\anaconda3\python.exe' -B 'D:\topic3\scripts\train.py' --help
```

此入口目前对应路径适配后的DeST阶段1，真实训练必须显式指定新的 `--run` 名称，使用 annotation15。它不是已经实现了全部基线的统一训练器。

## 历史与新实验

旧完整首轮的欧氏GMM、骨架TCN及参考对齐等已经做过，成绩按旧10fps映射保存；作者流形GMM尚未完整运行。旧普通/约束对齐也做过，但修正时钟后统一特征、模板、校准与解码的对照仍待开展。

历史JSON和报告中的原绝对路径、代码版本、旧“第二研究点”命名保留作追溯。运行路径以本README及configs/paths.json为准。legacy脚本可能仍有写回源目录的硬编码，不能直接执行。

工作副本已提供.gitignore。2026-10-14已初始化Git，当前分支main，origin为https://github.com/Marshall0214/topic3.git；提交范围遵守下文的Git同步规则。原始权重、缓存与大型预测保留在本机。

## Git同步范围

Git保存代码、配置、计划与必要的匿名划分ID。约2.5 GiB本地实验/归档资产、原始数据、权重、完整预测及作者源码副本按.gitignore排除，仍保留在本机。configs/upstream_sources.json记录作者源码地址和审计版本；全新克隆的依赖恢复与历史脚本限制见UPSTREAM_DEPENDENCIES.md。
