# K-Means

【[html完整版](https://fdujiag.github.io/PyML/UnSupervise/K_Means/)】

【[返回主仓](https://github.com/FDUJiaG/PyML)】

[TOC]

# 说明

 ## 文档

此为非监督学习中，K-Means 的说明文档，由于 github 公式限制，建议阅读【[html完整版](https://fdujiag.github.io/PyML/UnSupervise/K_Means/)】

**主要使用的包**

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import datasets
```

## 文件

| 文件           | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| KMeans_iris.py | KMeans 方法对 iris 数据集进行聚类的比较                      |
| kmeans.ipynb   | jupyter文档， Kmeans , Kmeans++ 及sklearn中官方KMeans 比较的示例<br />数据集包括 iris， boston， digits |
| kmeans_base.py | kmeans 基础算法                                              |
| kmeans_plus.py | kmeans++ 算法                                                |
| misc_utils.py  | 基础工具的函数库，包括距离计算，标签排序等                   |

# K-Means 介绍

## 前言

机器学习按照有无标签可以分为 **监督学习** 和 **非监督学习** 

监督学习里面的代表算法就是： SVM 、逻辑回归、决策树、各种集成算法等等

非监督学习主要的任务就是通过一定的规则，把相似的数据聚集到一起，简称聚类

K-Means 算法是在非监督学习比较容易理解的一个算法，也是聚类算法中最著名的算法

## K-Means 原理

K-Means 是典型的聚类算法，K-Means 算法中的 K 表示的是聚类为 K 个簇，Means 代表取每一个聚类中数据值的均值作为该簇的中心，或者称为质心，即用每一个的类的质心对该簇进行描述

# sklearn 的 K-Means 的使用

## K-Means 参数

- n_clusters : 聚类的个数k，default: 8
- init : 初始化的方式，default: [k-means++](##选取初始质心的位置)
- n_init : 运行 K-Means 的次数，最后取效果最好的一次， default: 10
- max_iter : 最大迭代次数， default: 300
- tol : 收敛的阈值，default: 1e-4
- n_jobs : 多线程运算，default = None，None代表一个线程，- 1 代表启用计算机的全部线程
- algorithm : 有 'auto',  'full' or 'elkan' 三种选择，'full' 就是我们传统的 K-Means 算法，'elkan' 是我们讲的[elkan K-Means](###elkan K-Means) 算法，默认的 'auto' 则会根据数据值是否是稀疏的，来决定如何选择 'full' 和 'elkan' ，一般数据是稠密的，那么就是 'elkan' ，否则就是 'full' ，一般来说建议直接用默认的 'auto' 

## K-Means 使用

```python
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_ 	# 输出原始数据的聚类后的标签值
>>> array([0, 0, 0, 1, 1, 1], dtype=int32)
kmeans.predict([[0, 0], [4, 4]]) 	# 根据已经建模好的数据，对新的数据进行预测
>>> array([0, 1], dtype=int32)
kmeans.cluster_centers_ 	# 输出两个质心的位置。
>>> array([[1., 2.],
       		 [4., 2.]])
```

KMeans 在 sklearn.cluster 的包里面，在 sklearn 里面都是使用 fit 函数进行聚类

顺便提一句，在 sklearn 中基本所有的模型的建模的函数都是 fit ，预测的函数都是 predict 

可以执行 Kmeans_iris.py 来进行鸢尾花数据分类的问题

<img src='img\kmeans_1.png' width=350><img src='img\kmeans_2.png' width=350>

<img src='img\kmeans_3.png' width=350><img src='img\kmeans_4.png' width=350>

1. 对数据用 $k=8$ 去聚类，因为数据本身只有 $3$ 类，所以聚类效果不好
2. 对数据用 $k=3$ 去聚类，效果不错
3. 还是用 $k=3$ 去聚类，但是改变初始化方式 init = random，n_init = 1，这样的随机初始化，并只运行 $1$ 次，最后的效果会不好
4. 最后一张图是数据本身的 label ，和右上相差不大

# 小结

K-Means的原理是很简单，但是我们仔细想想我们处理 K-Means 的思想和别的方法不太一样，先去猜想想要的结果，然后根据这个猜想去优化损失函数，再重新调整我们的猜想，一直重复这两个过程

其实这个猜想就是我们要求出的隐藏变量，优化损失函数的过程，就是最大化释然函数的过程，K-Means的算法就是一个 **EM 算法** 的过程



【[返回顶部](#线性回归)】

【[html完整版](https://fdujiag.github.io/PyML/UnSupervise/K_Means/)】

【[返回主仓](https://github.com/FDUJiaG/PyML)】