# 感知机

【[html完整版](https://fdujiag.github.io/PyML/Supervise/PLA/)】

【[返回主仓](https://github.com/FDUJiaG/PyML)】

# 说明

 ## 文档

此为监督学习中，感知机的说明文档。因为github公式限制，建议阅读【[html完整版](https://fdujiag.github.io/PyML/Supervise/PLA/)】

## 文件

| 文件               | 说明                                      |
| ------------------ | ----------------------------------------- |
| DM_Learner.py      | 感知机模型简单代码                        |
| perceptron.ipynb   | 感知机原始模型和对偶模型比较的jupyter文档 |
| perceptron_base.py | 感知机原始模型类                          |
| perceptron_dual.py | 感知机对偶模型类                          |
| utils_plot.py      | 绘制决策平面分离的代码                    |

## 前言

感知机是1957年，由Rosenblatt提出会，是**神经网络和支持向量机**的基础

# 感知机的原理

PLA全称是Perceptron Linear Algorithm，即线性感知机算法，属于一种最简单的感知机（Perceptron）模型

感知机是二分类的线性模型，其输入是实例的特征向量，输出的是事例的类别，分别是+1和-1，属于判别模型

假设训练数据集是线性可分的，感知机学习的目标是求得一个能够将训练数据集**正实例点和负实例点完全正确分开的分离超平面**。如果是非线性可分的数据，则最后无法获得超平面

<img src="https://note.youdao.com/yws/api/personal/file/WEB393a13769317374240437edb5e1d3b26?method=download&shareKey=4ee69663a8d0565951fda86a78c292a1" alt="image" style="zoom:50%;" />

# 训练过程

我们大概从下图看下感知机的训练过程。

线性可分的过程：

<img src="https://note.youdao.com/yws/api/personal/file/E28D8C8601F3472BAF94F8BC7F033318?method=download&shareKey=de7b09fdcefa8ab2504ef530825e0f11" style="zoom:90%;" />

线性不可分的过程：

<img src="https://note.youdao.com/yws/api/personal/file/84D4442B459B4B69B4B37B3539AAC117?method=download&shareKey=f6314f2f962a6f26b5a5250508704d30" style="zoom:90%;" />

# 小结

感知机算法是一个简单易懂的算法，它是很多算法的鼻祖，比如**支持向量机算法，神经网络与深度学习**。因此虽然它现在已经不是一个在实践中广泛运用的算法，还是值得好好的去研究一下。感知机算法对偶形式为什么在实际运用中比原始形式快，也值得好好去体会。

【[返回顶部](#感知机)】

【[返回主仓](https://github.com/FDUJiaG/PyML)】