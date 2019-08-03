# HeartAttack
在医院人造心脏瓣膜置换术后患者会存在感染风险。由于感染是极少出现的（1%？），因此患者感染数据集存在比较严重的比例不平衡问题。针对这一问题，本项目对原始数据进行了重采样，避免正例缺失问题，分别训练了ANN、RF和SVM三个模型，结果表明，数据重采样对预测F1值影响极大，结果可以参看f1_score.csv。本项目也是Risk Factor Analysis of Device-related Infections: Value of Re-sampling Method on the Real-World Imbalanced Dataset论文的部分内容。

由于隐私问题，如果需要原始数据，请联系邮件lzhtan@bjtu.edu.cn
