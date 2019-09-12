# 逻辑回归

【[html完整版](https://fdujiag.github.io/PyML/Supervise/Logistic/)】

【[返回主仓](https://github.com/FDUJiaG/PyML)】

[TOC]

# 说明

 ## 文档

此为监督学习中，线性回归的说明文档，由于github公式限制，建议阅读【[html完整版](https://fdujiag.github.io/PyML/Supervise/Logistic/)】

**主要使用的包**

```python
from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

## 文件

| 文件                            | 说明                                       |
| ------------------------------- | ------------------------------------------ |
| Logistic_Regression.py          | 逻辑回归实现代码                           |
| logistic_regression.ipynb       | 基于随机梯度下降和批梯度下降的线性回归代码 |
| LogisticRegressionClassifier.py | 逻辑回归方法类代码                         |
| data_generater.py               | 随机从load_iris选取数据集的代码            |

# 分类问题

对于监督学习中的分类问题，通常已知一些数据并知道它们各自属于什么类别，然后希望基于这些数据来判断新数据是属于什么类别的

比如已知一些症状特征和是否患有某种疾病的数据，基于这些数据来判断新的病人是否患病

再比如根据过去的垃圾邮件数据来判断新邮件是否为垃圾邮件

# 示例

数据集采用 sklearn 包中的官方鸢尾花数据集，有 $4$ 个特征及 $1$ 个标签

由于原始数据分 $3$ 类， 但 Logistic Regression 只能处理 $2$ 分类问题，所以暂时剔除标签值为 $-1$ 的样本

```python
from sklearn.datasets import load_iris
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    return data[:,:2], data[:,-1]
```

这里展示 SmallVagetable 用 sklearn 方法的结果，仅有 $1$ 个误分类点（$30\%$ 测试集）

<img src="img/Sklearn_Logistic.png" width=350/>

具体示例可详见 【[logistic_regression.ipynb]】

在自己的测试中，选取 $80\%$ 的样本点为训练集，选取 $20%$ 的点为测试集

```python
x_b shape (100, 4)
y_b shape (100,)
x_train shape (80, 4)
y_train shape (80,)
x_test shape (20, 4)
y_test shape (20,)
```

对于 sklearn 中 LogisticRegression 方法，可以给出theta以及预测效果 

```python
w [[-0.37088727 -1.40774288  2.12477048  0.93181512]]
b [-0.2335179]
theta [[-0.2335179  -0.37088727 -1.40774288  2.12477048  0.93181512]]
y_pred [1 0 1 1 1 0 1 1 1 0 1 0 0 0 1 1 0 0 1 0]
y_test [1 0 1 1 1 0 1 1 1 0 1 0 0 0 1 1 0 0 1 0]
```

而自行根据对 cost 函数的优化，同样可以得到类似的效果

```python
theta [-0.5191182  -0.71473815 -2.33185303  3.76260415  1.66776521]
loss 0.003273793693629498
y_pred [1 0 1 1 1 0 1 1 1 0 1 0 0 0 1 1 0 0 1 0]
y_test [1 0 1 1 1 0 1 1 1 0 1 0 0 0 1 1 0 0 1 0]
```

# 小结

逻辑回归假设数据服从**伯努利分布**，在线性回归的基础上，套了一个二分类的Sigmoid函数，使用**极大似然法**来推导出损失函数，用梯度下降法优化损失函数的一个**判别式的分类算法**

逻辑回归的优缺点有以下几点：

## 优点

1. 实现简单，广泛的应用于工业问题上
2. 训练速度较快，分类速度很快
3. 内存占用少
4. 便利的观测样本概率分数，可解释性强

## 缺点

1. 当特征空间很大时，逻辑回归的性能不是很好
2. 一般准确度不太高
3. 很难处理数据不平衡的问题



【[返回顶部](#线性回归)】

【[html完整版](https://fdujiag.github.io/PyML/Supervise/Logistic/)】

【[返回主仓](https://github.com/FDUJiaG/PyML)】