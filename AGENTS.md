# 第三点工作区约定

- 本工程根目录是 D:/topic3；本轮用户要求所有源目录文件不得修改或删除。
- D:/AQA/work/keraal_dest、D:/AQA/research/keraal_literature、D:/AQA/KeraalDataset、D:/action_compare/keraal_localization 均只读。原始数据 D:/AQA/dataset/keraal 也只读。
- legacy、upstream、runs/historical、reports/historical 和 manifests/historical 保存追溯副本。禁止直接运行含旧绝对路径的 legacy 脚本；从 scripts 的入口或新显式适配执行。
- 新结果写 topic3/runs 的新名称，不覆盖历史权重、预测、清单和划分。
- configs/paths.json 是当前路径入口。普通几何函数保留在 src/keraal_loc；topic3接口复用它们，不暗改原算法。
- 标注按 annotation15 映射；旧10fps成绩只作历史记录，不能混入正式排名。目录统一不代表实验协议修正已经完成。
- 错误部位原标注有四类，旧代码只用 Arms/Trunk；不得把旧两类实验称逐关节定位或四类定位。
- 修改模型或监督后须新登记配置与实验名；测试历史使用情况需披露。
- 不自动启动完整训练、重新推理测试集或安装环境；本次整理验收以已有契约检查、合成自检和源文件SHA校验为准。
