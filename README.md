@[TOC](组织病理学癌症检测Histopathologic Cancer Detection)
比赛链接:https://www.kaggle.com/c/histopathologic-cancer-detection

原文参考: https://www.kaggle.com/qitvision/a-complete-ml-pipeline-fast-ai

识别从较大的数字病理扫描中获取的小图像补片中的转移性癌症。此竞赛的数据是PatchCamelyon（PCam）基准数据集的略微修改版本（原始PCam数据集由于其概率抽样而包含重复图像，但是，在Kaggle上呈现的版本不包含重复项）。


PCam数据集将临床相关的转移检测任务打包成为，二分类任务，类似于CIFAR-10和MNIST。模型可以在几个小时内在单个GPU上轻松训练，并在Camelyon16肿瘤检测和整个幻灯片图像诊断任务中获得竞争分数。此外，任务难度和易处理性之间的较为平衡，可以学习研究基础机器学习模型不确定性和可解释性。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190301135832592.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Fycm9uX2hvdQ==,size_16,color_FFFFFF,t_70)
图1 染色后的图图片![在这里插入图片描述](https://img-blog.csdnimg.cn/2019030113580899.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Fycm9uX2hvdQ==,size_16,color_FFFFFF,t_70)
# 任务理解
## 问题的关键
**二值图像分类问题。** 确定96 x 96px数字组织病理学图像中转移的存在。一个关键的挑战是转移部分可能与大面积组织中的单个细胞一样小。
## 解决方案
**我们的评估指标是AUC。** 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190301142142398.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Fycm9uX2hvdQ==,size_16,color_FFFFFF,t_70#pic_center)

 **接收者操作特征曲线（Receiver Operating Characterisitc, ROC）** 十分适合于分类问题。
纵坐标为真正率$TPR=\frac{TP}{TP+FN}$表示正样本中预测为正的比值。
横坐标为假正率$FPR=\frac{FP}{FP+TN}$表示负样本中预测为正的比值。
理论上模型真正率为1，假正率为0达到最优。
ROC绘制是，通过修改分类阈值可以得到。根据数据量以及阈值的划分决定了ROC的平滑程度。
单独使用真正率或者假正率无法处理数据不均衡的问题，因此使用ROC（Receiver Operating Characteristic）曲线描述。
ROC曲线不能数值化，使用**ROC曲线下面积AUC(Area Under Roc Curve)** 作为评估指标。最佳解决方案将产生1的AUC，这意味着我们将正确地对所有阳性样本进行分类而不会产生任何误报。AUC等于分类器将随机选择的阳性实例排列为高于随机选择的阴性实例的概率，取值越大表示越有可能把正样本排在负样本之前。
详细解释：
>https://segmentfault.com/a/1190000010410634
https://blog.csdn.net/u013385925/article/details/80385873
## 领域的相关知识
组织病理学图像是用苏木精和伊红（H&E）染色的淋巴结的玻片显微镜图像。这种染色方法是医学诊断中应用最广泛的方法之一，它可以产生蓝色、紫色和红色。深蓝色苏木精与负电荷物质（如核酸和粉色伊红）结合，与正电荷物质（如氨基酸侧链）结合（大多数蛋白质）。典型的细胞核是蓝色的，而细胞质和胞外部分则是各种颜色的粉红色。

淋巴结是过滤淋巴系统中液体的小腺体，它们是乳腺癌最容易扩散的部位。淋巴结转移的组织学评估是确定TNM分类中乳腺癌分期的一部分，TNM分类是全球公认的癌症扩散程度分类标准。病理学家的诊断程序冗长费时，因为必须检查大面积的组织，很容易漏掉小的转移。
>**Useful links for background knowledge**
 [Patch Camelyon (PCam)](https://github.com/basveeling/pcam)
 [Hematoxylin and eosin staining of tissue and cell sections](https://www.ncbi.nlm.nih.gov/pubmed/21356829)
 [H&E-stained sentinel lymph node sections of breast cancer patients: the CAMELYON dataset](https://academic.oup.com/gigascience/article/7/6/giy065/5026175)
 [CAMELYON16 - background](https://camelyon16.grand-challenge.org/Background/)
 [CAMELYON17 - background](https://camelyon17.grand-challenge.org/Background/)
 [TNM classification](https://www.uicc.org/resources/tnm)




>[1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". arXiv:1806.03962
[2] Ehteshami Bejnordi et al. Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer. JAMA: The Journal of the American Medical Association, 318(22), 2199–2210. doi:jama.2017.14585



