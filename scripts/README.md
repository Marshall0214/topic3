# 当前入口与运行边界

本轮整理复制现有实现，未改模型公式、损失或选模规则，未开展训练。

|入口|用途|状态|
|---|---|---|
|run_checks.ps1|环境、既有检查、合成DeST自检、结果汇总、SHA校验|已执行通过|
|check_environment.py|读取现有Python与依赖版本、入口路径|可用，无安装行为|
|train.py --help|查看适配后的DeST阶段1参数|可用|
|train.py --selfcheck|合成数据前后向与时间/匹配检查|已执行通过，无优化器训练|
|train.py --run 新名称 ...|DeST阶段1训练，结果写根目录runs|路径已适配；本轮未训练，原方法失败/门槛仍保留|
|summarize_results.py|从保存result.json汇总开发指标|已验证，与原计数一致|
|verify_migration.py|核对源文件及原样副本的SHA和源mtime|已执行通过|
|organize_snapshot.py、setup_workspace.py|本次一次性整理工具|留作追溯，日常不需要重跑|

保存结果汇总（任意目录、一行）：

```powershell
& 'C:\Users\admin\anaconda3\python.exe' -B 'D:\topic3\scripts\summarize_results.py' --run historical/dest/stage1_clock15_local
```

R1/R2/R3、作者流形GMM及A-MAL/RF-DTAN当前没有新的统一CLI承诺；代码和既有结果已完整收录。下一阶段在现有公共函数上建立修正时钟后的统一基线，不能直接执行legacy中的旧绝对路径脚本。原完整首轮的B2_linear/B2_dtw是带参考的监督TCN，不等同纯几何线性/DTW基线。

环境建议：迁移检查使用现有base Python（CPU PyTorch）。dest_gpu环境缺少sklearn、numba等依赖，不作为本轮默认入口。旧环境锁文件保存在configs/legacy_principal_v1，实际检查版本见configs/environment.organized.json。没有新建或修改conda环境。
