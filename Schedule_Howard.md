#### 2024.9.18

- ML-Exercise1

#### 9.19

- ML-Exercise2

- ML-Exercise3

**正向传播算法**：生成预测

**反向传播算法**：用于学习，通过计算梯度更新网络的权重和偏置，以减少误差

- 9.2-9.4

#### 9.20

摆烂

#### 9.21

- ML-Exercise4-Ongoing
- 9.5-9.7

#### 9.22

- ML-Exercise4
- ML-Exercise5-Ongoing

#### 9.23

摆烂

#### 9.24

- ML-Exercise5-Ongoing
- 9.8

#### 9.25

- 10.1-10.3

#### 9.26

**偏差**:是衡量预测值与真实值的关系,是指预测值与真实值之间的差值

**方差**:是衡量预测值之间的关系,和真实值无关.也就是他们的离散程度

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202411210005207.png" alt="img" style="zoom:50%;" />

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202411210005140.png" alt="img" style="zoom: 50%;" /><img src="https://gitee.com/Goudabao/typora-images/raw/master/202411210005219.png" alt="img"  />

- 10.4-10.7

#### 9.27

- 11.1-11.4
- 11.5-12.1
- ML-Exercise5

**梯度**：即函数在某一点最大的方向导数，函数沿梯度方向函数有最大的变化率

​			它的方向与取得最大方向导数的方向一致，而它的模为方向导数的最大值

这里注意三点：
　1）梯度是一个向量，即有方向有大小；
　2）梯度的方向是最大方向导数的方向；
　3）梯度的值是最大方向导数的值



#### 10.3

- 12.2-12.5

#### 10.4

- 12.6

**支持向量机**：

​					选择参数C

​					选择核函数：
​										1线性核函数 No kernel（n特征数很大，m样本数较小）

​												预测"$y = 1$" if $\theta^ Tx\geq0$


​										2高斯核函数（n特征数很小，m样本数适中）
$$
f_i=\exp\left(-\frac{||x-l^{(i)}||^2}{2\sigma^2}\right)
$$
​													选择 $\sigma^2$

​																很大：小方差，大偏差

​																很小：大方差，小偏差

当特征数n很小，样本数m很大的时候可以创造新的特征，之后再使用logistic回归或者没有核函数的SVM

#### 10.5

- ML-Exercise6

#### 10.6

- 13.1-13.2

#### 10.7

- 13.3-13.5

- ML-Exercise7-Ongoing

#### 10.8

- 14.1-14.2
- ML-Exercise7-Ongoing

#### 10.9

- 14.3

#### 10.10

- 14.4-14.7
- ML-Exercise7

**主成分分析**，找到一个线或面，把数据投影到这个线或面上，并最小化平方投影误差(压缩、可视化)

在主成分分析（PCA）中，我们考虑保留原始数据的方差，是因为方差是衡量数据分散程度的一个重要统计量。方差大的方向表示数据在这个方向上的变化大，因此包含的信息也更多。PCA的目标是在降维的过程中尽可能保留这些变化大的方向，从而保留数据中最重要的信息。

不建议使用PCA来防止过拟合: PCA不区分特征的重要性，它平等地对待所有特征，通过线性组合生成新的特征。这可能导致一些不重要的特征被保留，而一些重要的特征被削弱或丢弃。

#### 10.11

- 15.1-15.4

#### 10.12

- 15.5-15.8

#### 10.13

- 17.1-17.2
- ML-Exercise8-Ongoing

#### 10.14

- 16.1-16.2

**协同过滤**![img](https://gitee.com/Goudabao/typora-images/raw/master/202411210003348.png)

#### 10.15

协同过滤模型不含偏置项，通常是为了捕捉用户和物品之间的关系，而不是为了拟合一个全局的偏好水平

#### 10.28

**鲁棒性**：指的是模型对于输入数据的健壮性，即模型在遇到各种不同的数据输入时，仍然能够保持高效的表现。一个鲁棒性强的模型能够在噪声、缺失数据或者其他异常情况下也能够准确地预测结果。

**泛化性**：则是指模型对于新数据的适应能力，即模型能否对于未在训练集中出现的数据进行准确的预测。一个具有很强泛化性的模型能够在不同的数据集上都表现出色，而不仅仅是在训练集上表现好。

鲁棒性关注的是模型对于已知情况的适应能力，而泛化性则关注的是模型对于未知情况的适应能力

#### 11.4

Tensorboard

```python
tensorboard --logdir=logs路径  #logs是上面指定在writer = SummaryWriter("logs")中指定的文件夹名，日志文件存储在此文件中
tensorboard --logdir==logs --port=XXXX # 可指定端口号
```



#### 11.5

##### 张量

在多维 Numpy 数组中，也叫张量（tensor）。一般来说，当前所有机器学习系统都使用张量作为基本数据结构。

张量这一概念的核心在于，它是一个数据容器。它包含的数据几乎总是数值数据，因此它是数字的容器。你可能对矩阵很熟悉，它是二维张量。张量是矩阵向任意维度的推广［注意，张量的维度（dimension）通常叫作轴（axis）］

###### 0. scalar标量 0D张量

仅包含一个数字的张量叫作标量（scalar，也叫标量张量、零维张量、0D 张量）。在 Numpy中，一个 float32 或  float64 的数字就是一个标量张量（或标量数组）。你可以用 ndim 属性来查看一个 Numpy 张量的轴的个数。标量张量有 0 个轴（ndim == 0）。张量轴的个数也叫作阶（rank）。下面是一个 Numpy 标量。

###### 1. vector 向量 1D张量

数字组成的数组叫作向量（vector）或一维张量（1D 张量）。一维张量只有一个轴。

###### 2. matrix 矩阵 2D张量

第一个轴上的元素叫作行（row），第二个轴上的元素叫作列（column）。

#### 11.6

Transforms

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202411210002042.png" alt="img" style="zoom: 33%;" />

#### 11.10

**.yaml**文件是一种使用 YAML（YAML Ain't Markup Language）格式的数据序列化格式。YAML 是一种人类可读的数据序列化标准，它被广泛用于配置文件、数据交换和存储等多种场景。

**.json**（JavaScript Object Notation）是一种轻量级的数据交换格式，它基于JavaScript中的对象表示方法。JSON以其简洁和易于阅读而广泛用于网络应用之间的数据传输。

#### 11.11-11.16

- 小土堆PyTorch深度学习--Ongoing

#### 11.17

##### LDA.fit_transform()

- **用途**：这个方法结合了拟合（fit）和转换（transform）两个步骤。它首先使用训练数据来估计LDA模型参数（例如，计算类内散度矩阵和类间散度矩阵），然后使用这些参数将训练数据转换到低维空间。
- **功能**：`fit_transform()` 方法返回的是降维后的训练数据。
- **参数**：需要训练数据 `X` 和对应的标签 `y`。
- **返回值**：返回的是降维后的数据矩阵。

##### LDA.transform()

- **用途**：这个方法仅用于转换数据，它使用已经拟合好的LDA模型参数来转换新数据（通常是测试数据）到低维空间。
- **功能**：`transform()` 方法返回的是降维后的新数据。
- **参数**：需要新数据 `X`，这些数据不需要对应的标签。

#### 11.18

- 小土堆PyTorch深度学习--Ongoing

- 李沐深度学习01-03

#### 11.19

- 李沐深度学习04

#### 11.20

- 小土堆PyTorch深度学习--Ongoing

- 李沐深度学习05 线性代数



<img src="https://gitee.com/Goudabao/typora-images/raw/master/202411210008641.png" alt="image-20241121000801575" style="zoom:50%;" />

Padding：在输入数据（通常是图像）的边界周围添加一定数量的零值像素。

Stride：卷积核在输入数据上滑动时的步长。



<img src="https://gitee.com/Goudabao/typora-images/raw/master/202411210127031.png" alt="image-20241121012751788" style="zoom: 33%;" />

#### 11.22

- 李沐深度学习06-07 矩阵计算 自动求导

矩阵求导：按列优先求导，然后按行优先放到结果矩阵

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202411222244939.png" alt="image-20241122224421833" style="zoom:50%;" />

#### 11.23

摆烂

#### 11.24

- 李沐深度学习08 线性回归+基础优化算法

#### 11.25

摆烂

#### 11.26

- 李沐深度学习09 Softmax回归+损失函数+图片分类数据

- 吴恩达DeepLearning 3.1-3.5 Softmax

**Logistic**函数是一种常用的S型（**sigmoid**）函数,能够将任意实数值映射到0和1之间,它的S型特性和输出范围使其在处理二分类问题时非常有效。
$$
f(x)=\frac{1}{1+e^{-x}}
$$
其Loss函数是Binary Cross Entropy Loss，也称为**对数损失**（Log Loss）。对于单个样本，Logistic回归的Loss函数定义为：
$$
\ell=-[y\log(\hat{y})+(1-y)\log(1-\hat{y})]
$$

**Softmax**函数是一种在机器学习和深度学习中常用的激活函数，特别是在处理分类问题时。它将一个实数向量转换为一个概率分布，使得向量中的每个元素都在0到1之间，并且所有元素的和为1。这使得Softmax函数非常适合用来表示多个类别的概率。
$$
\mathrm{Softmax}(\mathbf{z})_i=\frac{e^{z_i}}{\sum_{j=1}^ke^{z_j}}
$$
其Loss函数是Cross-Entropy Loss，也称为**交叉熵损失**。对于单个样本，Softmax回归的Loss函数定义为：
$$
l(\mathbf{y},\hat{\mathbf{y}})=-\sum_{j=1}^qy_j\log\hat{y}_j
$$

**相对熵**又称为[KL散度](https://so.csdn.net/so/search?q=KL散度&spm=1001.2101.3001.7020)（Kullback–Leibler divergence），用来描述两个概率分布的差异性。假设有对同一变量*x*预测的匹配分布*q(x)*和目标分布*p*(*x)*两个概率分布,那么两者之间的相对熵可由以下定义：
$$
D_{KL}\left(p\|q\right)=\sum_{i=1}^Np\left(x_i\right)\log\left(\frac{p\left(x_i\right)}{q\left(x_i\right)}\right)
$$


两个分布差异越大，KL散度越大。实际应用需要两个分布尽可能的相等，于是就需要KL散度尽可能的小。

**交叉熵**是信息熵论中的概念，它原本是用来估算平均编码长度的。在深度学习中，可以看作通过匹配分布*q(x)*表示目标分布*p*(*x)*的困难程度。其表达式为：
$$
H(p,q)=\sum_{i=1}^np\left(x_i\right)\log\frac{1}{q\left(x_i\right)}=-\sum_{i=1}^np\left(x_i\right)\log q\left(x_i\right)
$$
交叉熵刻画的是两个概率分布的距离，也就是说交叉熵值越小(相对熵的值越小),两个概率分布越接近。

两者的关系：
$$
D_{KL}\left(p\|q\right)=\sum_{i=1}^{N}p\left(x_{i}\right)\log\left(\frac{p\left(x_{i}\right)}{q\left(x_{i}\right)}\right)=\sum_{i=1}^{N}p\left(x_{i}\right)\log p\left(x_{i}\right)-\sum_{i=1}^{N}p\left(x_{i}\right)\log q\left(x_{i}\right)=-H(p)+H(p,q)\geq0
$$
当且仅当*q(x)*=*p*(*x)*时，取最小值0

#### 11.27

摆烂

#### 11.28

- 小土堆PyTorch深度学习-卷积层

TensorFlow 或 PyTorch，图像数据的形状为 (批次大小, 高度, 宽度, 通道数)

如果一批包含 32 张 224x224 像素的彩色图像，那么它们的形状是 (32, 224, 224, 3)。

如果一批包含 32 张 224x224 像素的灰度图像，那么它们的形状是 (32, 224, 224)。

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202411290010957.png" alt="image-20241129001008815" style="zoom:50%;" />

#### 11.29

- 李沐深度学习10 感知机

激活函数：为了使神经网络模型具有非线性特性

**Sigmod函数**：$f\left(x\right)=\frac{1}{1+e^{-x}}$

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202411300050642.png" alt="image-20241130005059585" style="zoom:50%;" />

**tanh函数**：$f\left(x\right)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202411300052230.png" alt="image-20241130005230203" style="zoom: 67%;" />

**ReLU函数**：$f(x)=\max(0,x)$  *不同激活函数本质没有太多区别，选ReLU就可以*

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202411300053515.png" alt="image-20241130005322475" style="zoom: 50%;" />



*Q选择增加隐藏层的层数，而不是增加神经元的个数  Why?*

*A因为增加隐藏层（深度学习）更容易训练，而增加神经元的个数（增加宽度）更容易导致过拟合*



#### 11.30

- 李沐深度学习11 模型选择+过拟合和欠拟合

**训练数据集**：训练模型参数，如神经网络中的权重和偏置。

**验证数据集**：选择模型超参数，如学习率、正则化系数、网络层的数量和大小等。

*Q k折交叉验证的目的是确定超参数，之后再使用该超参数训练一遍数据吗？*

*A 有n种做法：1 如题；2 挑选出性能最好的一折所得模型参数；3 做预测时，将k个模型全部拿出来，将数据在k个模型中都预测一次，将结果取均值*

#### 12.1

- 李沐深度学习12 权重衰退

**权重衰退：**在模型原有的损失函数基础上，加上一个额外的惩罚项，这个惩罚项通常是模型权重向量的L2范数乘以一个正则化参数*λ*，这种做法鼓励模型学习到更小的权重值

参数更新法则：

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412020026161.png" alt="image-20241202002609004" style="zoom: 25%;" />

weight_decay一般选择$e^{-2}e^{-3}e^{-4}$



*Q 为什么需要通过惩罚项$L1$h或$L2$将$w$变小，如果最优解的$w$就是比较大，权重衰减会不会起到反作用？*

*A 如果最优解的权重本来就较大，权重衰减可能会起到反作用，因为它会倾向于减小权重。然而，实际上很难达到完全的最优解，因为存在噪声。正则化的目的是在一定程度上减少模型对噪声的敏感性，将结果拉向更接近最优解的方向*



#### 12.2

- 李沐深度学习13 丢弃法

**丢弃法Dropout** 将一些输出项随机置0来控制模型复杂度, 提高模型的泛化能力; 常作用在多层感知机的隐藏层输出上; 丢弃概率是控制模型复杂度的超参数。

它对每个元素进行如下扰动：
$$
\left.x_i^{\prime}=\left\{
\begin{array}
{cc}0 & \text{with probablity} \quad p \\
\frac{x_i}{1-p} & \mathrm{otherise}
\end{array}\right.\right.
$$
即加入噪音后的$x^{\prime}$有: $\mathbf{E}[x^{\prime}]=x$ （p通常取0.9 0.5 0.1）



使用有噪声的数据等价于Tikhonov正则，而丢弃法是在层之间加入噪声

训练时丢弃，预测时不丢弃，每迭代一次，就随机丢弃一次

#### 12.3

摆烂

#### 12.4

- 李沐深度学习14 数值稳定性 + 模型初始化和激活函数

**矩阵求导**

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412042044892.png" alt="image-20241204204420696" style="zoom:33%;" />

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412042046208.png" alt="image-20241204204641084" style="zoom:33%;" />

特例1：（使用分母布局，若使用分子布局，则不带转置）

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412042047665.png" alt="image-20241204204740565" style="zoom:33%;" />

特例2：

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412042050942.png" alt="image-20241204205024823" style="zoom:33%;" />

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412042051457.png" alt="image-20241204205100392" style="zoom:33%;" />

**Xavier 初始化**通过设置权重的初始值，使得每一层的输出方差与输入的方差相等（为了防止梯度爆炸/梯度消失）

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412050023249.png" alt="image-20241205002308124" style="zoom: 33%;" />

其中，$n_{t-1}$和$n_t$分别表示第t层输入和输出的维度，$\gamma_t$表示第$t$层权重的方差

检查激活函数在0附近是否满足$\sigma(x)=x$

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412050038751.png" alt="image-20241205003817609" style="zoom: 33%;" />



*Q强制使得每一层的输出特征均值为0，方差为1，是不是损失了网络的表达能力，改变了数据的特征？会降低学到的模型的准确率？*

*A并不会*



*Q权重是在每一个epoch结束之后更新的吗？*

*A并不是，更新伴随着每一个batch，一个epoch之后已经更新很多次了*



#### 12.5

- 李沐深度学习15 实战：Kaggle房价预测 + 课程竞赛

#### 12.6

- 李沐深度学习16 PyTorch 神经网络基础 --Ongoing

  **全连接层**：

  - 全连接层是神经网络中的一种层类型，其中每个输入节点与每个输出节点之间都有一个连接。
  - 它通过线性变换（加权求和加上偏置）将输入特征映射到输出空间。
  - 在 PyTorch 中，全连接层通常由 `nn.Linear` 实现。

  **隐藏层**：

  - 隐藏层是指神经网络中介于输入层和输出层之间的层。
  - 隐藏层的主要作用是提取和转换特征，以便更好地进行预测或分类。
  - 隐藏层可以由多种类型的层组成，包括全连接层、卷积层、循环层等。

  **关系**：

  - 全连接层可以用来构建隐藏层。例如，一个包含全连接层的隐藏层可以通过 `nn.Linear` 实现。
  - 隐藏层通常包含激活函数（如 ReLU、Sigmoid、Tanh 等），以引入非线性变换，从而增强模型的表达能力。

**控制流**：指程序执行过程中控制语句（如条件语句和循环语句）的执行顺序。

**Sequential对象**nn.Sequential是PyTorch 中的一个容器模块，用于按顺序包含和执行一系列子模块。它提供了一种简洁的方式来构建神经网络模型，特别适用于那些层次结构简单、按顺序执行的模型。子模块可以是任何 nn.Module的实例，例如全连接层、卷积层、激活函数等。

```python
# 定义一个简单的神经网络模型
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16)
)

# 创建一个随机输入张量
X = torch.rand(5, 20)

# 进行前向传播
output = model(X)
print(output)
```

在这个示例中：

- nn.Sequential包含了三个全连接层和两个 ReLU 激活函数。
- 输入张量 X 依次通过每个子模块进行前向传播，最终得到输出。

#### 12.7

- 李沐深度学习16 PyTorch 神经网络基础
- 李沐深度学习17 使用和购买GPU

同一块GPU上面的数据才可以进行运算



*Q如何解决类别变量转换为伪变量的时候发生内存爆炸的问题*

*A使用稀疏矩阵*



#### 12.8

- 李沐深度学习18 预测房价竞赛总结

#### 12.9

- 李沐深度学习19 卷积层

**平移不变性**：模型对于输入数据的平移（位置变化）具有不变性。也就是说，如果输入图像中的对象发生平移，模型仍然能够识别出对象。

**局部性**：指的是卷积操作只关注输入数据的局部区域，而不是全局范围。每个滤波器只对输入图像的一个小块区域进行处理。

#### 12.10

- 李沐深度学习19 卷积层

核矩阵的大小是超参数

卷积就是一个特殊的MLP



*Q为什么不把卷积核的尺寸放大*

*A和神经网络类似，窄的深的神经网络效果要比宽的浅的效果好，越小的卷积核，需要学习的参数就越小，进而越高效*



*Q全连接层和MLP有什么区别和联系*

*A全连接层是单一的层，而MLP是由多个全连接层组成的网格结构；*

*MLP中的每个层都是全连接层，这意味着MLP实际上是由多个全连接层组成的*

#### 12.11

- 李沐深度学习20 卷积层里的填充和步幅
- 李沐深度学习21 卷积层里的多输入多输出通道
- 吴恩达DeepLearning 1.6 三维卷积

**填充**

填充$p_h$行和$p_w$列，输出形状为：
$$
\quad (n_h - k_h + p_h + 1) \times (n_w - k_w + p_w + 1)
$$
通常取：$\quad p_h = k_h - 1, \quad p_w = k_w - 1$

​	当$k_h$为奇数时：在上下两侧填充$\frac{p_h}{2}$

​	当$k_h$为偶数时（不常见）：在上侧填充$\left\lfloor \frac{p_h}{2}\right\rfloor$(下取整)，在下侧填充$ \left\lceil \frac{p_h}{2}\right\rceil$(上取整 ) (反过来也可以)

**步幅**

给定高度 $s_h$ 和宽度 $s_w$ 的步幅，输出形状是
$$
\left\lfloor \frac{(n_h - k_h + p_h + s_h)}{s_h} \right\rfloor \times \left\lfloor \frac{(n_w - k_w + p_w + s_w)}{s_w} \right\rfloor
$$
如果 $p_h = k_h - 1$， $p_w = k_w - 1$  （此处的 $p_h$等于代码中2倍的padding）
$$
\left\lfloor \frac{(n_h + s_h - 1)}{s_h} \right\rfloor \times \left\lfloor \frac{(n_w + s_w - 1)}{s_w} \right\rfloor
$$
如果输入高度和宽度可以被步幅整除
$$
\left( \frac{n_h}{s_h} \right) \times \left( \frac{n_w}{s_w} \right)
$$
**多输入通道**中每个通道都有一个卷积核，结果是所有通道卷积结果的和

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412111942806.png" alt="image-20241211194156689" style="zoom: 67%;" />

输出通道是卷积层的超参数（输入通道是上一层的）

![image-20241211204707278](https://gitee.com/Goudabao/typora-images/raw/master/202412112047430.png)

输入通道数等于卷积核通道数，卷积核数等于输出通道数

1x1的卷积核相当于一个全连接层



*Q超参数：核大小、填充、步幅的影响重要程度如何排序？*

*A填充一般固定不变*

*步幅一般为一，当计算过于复杂可改为二*

*核大小通常为最关键*



*Q卷积中的bias对结果的影响大吗 ？*

*A几乎不会有影响*



#### 12.12

- 李沐深度学习22 池化层 
- 李沐深度学习23 经典卷积神经网络 LeNet --Ongoing

**池化层**：通过降低特征图的维度，池化层有助于减少模型的复杂度，从而降低过拟合的风险

**最大池化层**：每个窗口中最强的模式信号

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412140117324.png" alt="image-20241214011717178" style="zoom: 33%;" />



**平均池化层**：将最大池化层中的“最大”替换成“平均”

可以缓解卷积层对位置的敏感性

位于卷积层之后

没有可学习的参数，但和卷积层一样有窗口大小、填充、和步幅作为超参数

输出通道数=输入通道数

**LeNet**先使用卷积层来学习图片空间信息，再使用全连接层转换到类别空间

![image-20241214020244647](https://gitee.com/Goudabao/typora-images/raw/master/202412140202755.png)



#### 12.13

- 李沐深度学习23 经典卷积神经网络 LeNet 

#### 12.14

- 李沐深度学习24  深度卷积神经网络 AlexNet --Ongoing

#### 12.15

- kaggle
- 李沐深度学习24  深度卷积神经网络 AlexNet 

相比于LeNet，AlexNet新加入了丢弃法、ReLU，最大池化层和数据增强

AlexNet架构

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412151746048.png" alt="image-20241215174629893" style="zoom: 33%;" />

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412151748486.png" alt="image-20241215174809341" style="zoom:33%;" />



*Q为什么池化层要放在第一个和最后一个卷积层后面？*

*A玄学*



*除了nlp和cv之外，推荐系统中也比较流行深度学习*



*Q网络要求输入的size是固定的，实际使用的时候图片不一定是要求的size，如果强行resize成网络要求的size，会不会使最后的效果变差?*

*A实际使用中对图片保持高宽比进行缩放，然后裁剪*



#### 12.16

- 李沐深度学习 使用块的网络VGG
- 李沐深度学习 网络中的网络NiN

**VGG**使用可重复的卷积来构建深度卷积神经网络,不同的卷积块个数和超参数可得到不同复杂度的变种

VGG块使用3X3卷积（padding = 1） n层m通道  2X2MaxPool（stride = 2）

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412152334896.png" alt="image-20241215233442785" style="zoom: 50%;" />

VGG架构

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412152336297.png" alt="image-20241215233657195" style="zoom: 50%;" />

其中，Dense为全连接层

#### 12.17

- 李沐深度学习26 网络中的网络NiN

**NiN块**：一个卷积层后跟两个全连接层（对每个像素增加了非线性性）*（NiN现在实际使用较少）*

![image-20241217223640148](https://gitee.com/Goudabao/typora-images/raw/master/202412172236256.png)

其中，1X1卷积层相当于全连接层

NiN架构交替使用NiN块和步幅为2的最大池化层，逐步减小高宽和增大通道数。最后使用全局平均池化层替代VGG和AlexNet中的全连接层得到输出，其中输入通道数是类别数

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412172242105.png" alt="image-20241217224218974" style="zoom:50%;" />

Softmax函数已经在损失函数中体现，所以不放在网络上

优势：不容易过拟合，更少的参数个数

#### 12.18

- 李沐深度学习27 含并行连结的网络 GoogLeNet / Inception V3

和LeNet网络并没有任何关系

**Inception块**用4条有不同超参数的卷积层和池化层的路来抽取不同的信息，计算复杂度低

![image-20241218224013619](https://gitee.com/Goudabao/typora-images/raw/master/202412182240691.png)

高宽减半为一个Stage

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412182249276.png" alt="image-20241218224905174" style="zoom: 50%;" />

#### 12.19

- 李沐深度学习28 批量归一化--Ongoing

#### 12.20-12.21

- 摆烂

#### 12.22

- 李沐深度学习28 批量归一化
- 吴恩达DeepLearning  3.5-3.7 将Batch Norm 拟合进神经网络

**批量归一化**：固定小批量中的均值和方差，然后学习出适合的偏移和缩放。

应用于卷积层/全连接层输出后，激活函数前。对于全连接层，作用在特征维，对于卷积层，作用在通道维。 

可以加速收敛速度，但是一般不改变模型精度。

类似于丢弃法dropout有轻微正则化的效果，可以同时使用

固定小批量里的均值和方差
$$
\mu_B = \frac{1}{|B|} \sum_{i \in B} x_i \\
\sigma_B^2 = \frac{1}{|B|} \sum_{i \in B} (x_i - \mu_B)^2 + \epsilon
$$
然后进行额外的调整，其中$\gamma, \beta$是可学习的参数,分别对应方差的均值
$$
x_{i+1} = \gamma \frac{x_i - \mu_B}{\sigma_B} + \beta
$$
<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412221656193.png" alt="image-20241222165637945" style="zoom:50%;" />

*Q xavier中也有提到normalization，和BN有什么区别？*

*A 归一化是初始化权重，BN是使得每一层数据分布更稳定*



#### 12.23

- 李沐深度学习29 残差网络 ResNet
- 李沐深度学习29.2 ResNet为什么能训练出1000层的模型 
- 吴恩达DeepLearning 2.3 指数加权平均

更复杂的模型需要包括之前的模型，才能使得结果更接近于最优

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412231810815.png" alt="image-20241223181022634" style="zoom:33%;" />

**残差块：**串联一个层来改变函数类，$f(x) = x + g(x)$, 和GoogLenet不同的是，GoogLenet直接合并通道

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412231819829.png" alt="image-20241223181933710" style="zoom:33%;" />

**ResNet架构**

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412231949468.png" alt="image-20241223194904381" style="zoom: 50%;" />



乘法变加法来解决梯度消失问题

普通加层中，最后的拟合能力较强的层如全连接层，对底层权重求导所得的数值较小也就是和真实值之间的误差较小，会导致梯度趋于0，更新底层的系数会变得困难（全连接层更耗费内存和计算）

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412232019009.png" alt="image-20241223201909888" style="zoom:33%;" />



*Q残差体现在什么地方？*

*A对于$f(x) = x + g(x)$来说，先训练下层的部分x，再根据不同层之间Loss的残差逐级去训练下一层*



#### 12.24

- 吴恩达DeepLearning 2.4-2.5 指数加权平均
- 吴恩达DeepLearning 2.6 Momentum梯度下降法

**指数加权平均**：$v_t=\beta v_{t-1}+(1-\beta)\theta_t$, 其中，$\beta$越大，越平滑

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412242201049.png" alt="image-20241224220153802" style="zoom:33%;" />

如果关注初始时期的偏差，可以加入**偏差修正**：$\frac {v_t}{1-\beta_t}$实现紫色曲线到绿色曲线（一般不使用）

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412242223526.png" alt="image-20241224222307433" style="zoom: 50%;" />

**Momentum梯度下降：**计算梯度的指数加权平均数，并利用该梯度更新权重。（将曲线拉平）

它的运行速度几乎总是快于标准梯度下降

$v_{dw}=\beta v_{dw}+(1-\beta)dw$

$v_{db}=\beta v_{db}+(1-\beta )d_b$

使用$v_{dw}$代替$w$：

$w=w-\alpha v_{dw}, b = b-\alpha v_{db}$

其中，超参数为学习率$\alpha$和参数$\beta$

（$\beta=0.9$比较常见，代表加权平均前十次的迭代，0.98为前50次）

#### 12.25

- 李沐深度学习 31 深度学习硬件：CPU 和 GPU（略）
-  吴恩达DeepLearning Mini-batch梯度下降

模型大小和复杂度并不一定正比

**Mini-batch**将整个训练数据集分成多个较小的子集，每个子集包含一定数量的样本

对于一组50w个样本的数据集，将其分为5000个Mini-batch, 每个batch里面1000个样本

batch → 一个epoch做一次梯度下降

Mini-batch →  一个epoch做5000次梯度下降

如果 Minibatch size = 50w，对应batch（可能会超过内存容量）

如果 Minibatch size = 1，对应随机梯度下降SVG（永远不会收敛，一直在最小值附近波动；效率不高）

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202601241743847.png" alt="image-20241225190937356" style="zoom:33%;" />

batch的每一次迭代对应代价都应该下降。但是Mini-batch在整体趋势上，代价仍然是下降的，只是下降的过程中会有更多的噪声

#### 12.26

- 李沐深度学习32 深度学习硬件：TPU和其他（略）

#### 12.27

- 李沐深度学习33 单机多卡并行
- 李沐深度学习34 多GPU训练实现

**数据并行**：

将小批量数据分成n块，每个GPU获得完整的模型参数，然后各自计算其分配到的数据块的梯度。

这种方法通常能够提供更好的性能，因为它允许多个GPU同时工作在不同的数据块上，从而加快了训练过程。

**模型并行**：

将模型分成n块，每个GPU负责模型的一部分，计算其负责部分的前向传播和反向传播结果。

这种方法通常用于模型太大，以至于单个GPU无法容纳的情况。通过将模型的不同部分分配给不同的GPU，可以训练更大的模型。



*Q对于精度来讲，batch size = 1是一种最好的情况吗？*

*A或许是的*



#### 12.28

- 李沐深度学习 35 分布式训练



*Q为什么batchsize越大，训练的有效性反而会下降？*

*A batchsize达到一定程度，每个batch内的样本的多样性不会比之前有多大增长，对梯度的贡献也不会比之前的batch大多少，但是大的batchsize会带来更多的训练时间，就造成了训练有效性下降。（训练效率）假设总共有1_0000个样本 1. 如果一个batch是100, 那么一个epoch可以迭代100次梯度; 2. 如果一个batch是1000, 那么一个epoch只能迭代10次梯度, 如果想要收敛, 意味着需要更多epoch.*



#### 12.29

- 李沐深度学习 36 数据增广

**数据增广**：通过对现有数据进行变换生成新数据来增加数据集的大小和多样性，提高模型的泛化能力，减少过拟合的风险。（在线生成，而不是本地）



*Q如何理解多样性增加，但是分布不变？*

*A可以认为，增广使得均值不变，方差变大*



#### 12.30

- 李沐深度学习 37 微调(迁移学习)

一个神经网络一般可以分为两个部分：

（1）特征抽取，将原始像素编程容易线性分割的特征

（2）分类，使用线性分类器，如全连接层的Softmax回归

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412301147076.png" alt="image-20241230114737954" style="zoom:33%;" />

特征提取部分可能仍然对新数据集适用，于是可以：

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202412301154125.png" alt="image-20241230115432985" style="zoom: 33%;" />

可以使用更小的学习率；使用更少的数据迭代

当原数据集远复杂于目标数据集时，微调效果更好

最后一层的可以使用较大的学习率，如learning rate*10，其他层用较小的learning rate 



*Q如果源数据集和目标数据集差异很大，微调的效果会下降吗，例如imagenet上的模型应用到医疗影像分类？*

*A会的*



#### 12.31 

- 李沐深度学习 39 图像分类 CIFAR -10 - Ongoing

#### 2025.1.1

- 李沐深度学习 39 图像分类 CIFAR -10

**学习率调度器**（Learning Rate Scheduler）是机器学习中用来动态调整学习率的工具，它在训练过程中根据预定义的策略自动调整学习率，以提高训练效率和模型性能。余弦退火、指数衰减、指定步数、固定步数



*Q深度学习的损失函数一般是非凸的吗？*

*A损失函数是凸函数，但是神经网络一般都是非凸的，所以神经网络＋损失函数就是非凸的。从实用角度讲，凸函数一般没用*



#### 1.2

- 李沐深度学习 40 狗的品种识别（ImageNet Dogs）



*Q在test_transform中，为什么要resize到256，再在中心裁剪224，相比于直接把图片resize到224有什么好处？*

*A历史遗留问题，没有特殊作用*



#### 1.3

- 李沐深度学习 41 物品检测和数据集

#### 1.4

摆烂

#### 1.5

- 李沐深度学习 42 锚框 -Ongoing

**边缘框**bonding box：真实的位置

**锚框**anchor box：算法预测的位置

**IoU**（Intersection of Union）:用来计算两个框之间的相似度

给定两个集合A和B, 有Jaccard相关系数：$$J(A,B)=\frac{|A \cap B|}{|A \cup B|}$$

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202501051742305.png" alt="image-20250105174257209" style="zoom:33%;" />

每个锚框都是一个训练样本，要么标注成背景，要么并联上一个真实的边缘框。

我们可能会生成大量的锚框，这会导致大量的负类样本

**赋予锚框标号**的一种常见用法

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202501051753244.png" alt="image-20250105175351120" style="zoom:30%;" />

**非极大值抑制（NMS）输出**

选中非背景类的预测，去掉所有其他和IoU大于$\theta$的预测，重复该过程，直到所有预测要么被选中，要么被去掉

#### 1.6-1.7

#### 1.8

- 李沐深度学习 42 锚框

*锚框是一个regression问题，所以没有置信度这个说法*

#### 1.9

- 李沐深度学习 43 树叶分类竞赛技术总结

工业界会花更很多的时间如80%在数据质量上，而不是调模型

#### 1.10

- 李沐深度学习 44 物体检测算法：R-CNN，SSD，YOLO

**兴趣区域（ROI）池化层**：给定一个锚框，均匀分割为nxm块，输出每块的最大值，无论锚框多大，总是输出nm个值（可能会有分割不均匀的现象）

目的是为了解决最终得到的锚框大小不一致的问题

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202501101720777.png" alt="image-20250110172001589" style="zoom:33%;" />



**RCNN：**生成锚框之后，对每一个锚框CNN提取特征

**Fast RCNN**：对图片CNN提取特征，之后再将生成锚框映射到FeatureMap上

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202501102056513.png" alt="image-20250110205612436" style="zoom: 50%;" />

**Faster RCNN：**使用趋于提议网络替代启发式搜索来获得更好的锚框

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202501102101457.png" alt="image-20250110210148371" style="zoom:50%;" />

**Mask RCNN：**如果有像素级的标号，使用FCN来利用这些信息

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202501102104894.png" alt="image-20250110210432797" style="zoom:40%;" />

其中，RoI pooling →RoI align， RoI align没有暴力取整，而是保留了浮点

Faster RCNN和Mask RCNN是在追求精度场景下的常用算法	

#### 1.11

- 李沐深度学习 45 SSD实现-Ongoing

#### 1.12

- 李沐深度学习 45 SSD实现-Ongoing

#### 1.13

- 李沐深度学习 45 SSD实现-Ongoing

**单发多检测框（Single Shot MultiBox Detector, SSD）**的五个stage:

```python
def get_blk(i):
	if i == 0:
		blk = base_net() # 从原始图片提取特征到第一次feature map做锚框，中间的这一段为base_net. 其中包括三个down_sample_blk
	elif i == 1:
		blk = down_sample_blk(64, 128) # 高宽减半块
	elif i == 4:
		blk = nn.AdaptiveMaxPool2d((1,1))
	else:
		blk = down_sample_blk(128, 128)
	return blk
```

**锚框（Anchor Box）**：锚框是预定义的一组框，覆盖图像中的不同位置、尺度和宽高比。它们是固定的，不会随着训练过程而改变。

**偏移量（Offsets）**：偏移量是相对于锚框的调整参数，用于修正锚框的位置和大小，使其更好地拟合目标物体。

在目标检测任务中，通常需要同时预测两个方面的信息：锚框的边界框坐标（即位置回归）和锚框的类别（即分类）。这两个任务分别由不同的预测器完成：

1. 边界框回归（Bounding Box Regression）：

   ​	预测每个锚框的四个边界框坐标参数（偏移量），即 `(x, y, width, height)`。

   ​	这些参数用于调整锚框的位置和大小，使其更好地拟合目标物体。

2. 类别分类（Class Classification）：

   ​	预测每个锚框的类别，通常是 `number_class + 1` 个类别，其中 `number_class` 是目标类别的数量，`+1` 表示背景类。

#### 1.14

- 李沐深度学习 45 SSD实现-Ongoing

#### 1.15

- 李沐深度学习 45 SSD实现



*对每个像素做锚框，这里的像素指的是feature map的像素而不是输入图片的像素*



#### 1.16

- 李沐深度学习46 语义分割-Ongoing

**语义分割**：它重点关注于如何将图像分割成属于不同语义类别的区域。与目标检测不同，语义分割可以识别并理解图像中每一个像素的内容，其语义区域的标注和预测是像素级的

#### 1.17

- 李沐深度学习46 语义分割

**PyTorch**：图像数据的维度顺序通常是 `(C, H, W)`，即通道数（Channels）在前，高度（Height）和宽度（Width）在后。

**Matplotlib**：图像数据的维度顺序通常是 `(H, W, C)`，即高度和宽度在前，通道数在后。



#### 1.18

- 李沐深度学习47 转置卷积

**转置卷积**可以增大输入的高宽

![image-20250119011653995](https://gitee.com/Goudabao/typora-images/raw/master/202501190117117.png)

**卷积和转置卷积**的另一种理解形式：

![img](https://gitee.com/Goudabao/typora-images/raw/master/202501190140394.png)

![img](https://gitee.com/Goudabao/typora-images/raw/master/202501190141818.png)

填充$p_h$行和$p_w$列，给定高度 $s_h$ 和宽度 $s_w$ 的步幅，输出形状为：
$$
\left\lceil n_hs_h + k_h - p_h - s_h \right\rceil \times \left\lceil n_ws_w + k_w - p_w - s_w \right\rceil
$$
一般取$k - p - s=0$，此时$k$即为图像放大的倍数



下面是两种卷积的示意图, 它们的具体参数为 $s=1;p=0;k=3$

**卷积**

​																				<img src="https://gitee.com/Goudabao/typora-images/raw/master/202501192150365.gif" alt="img" style="zoom: 50%;" />

**转置卷积**

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202501192158896.gif" alt="img" style="zoom: 50%;" />

计算流程：

（1）在输入特征图元素**之间**填充$s-1$的行和列的0元素；

（2）在输入特征图的**四周**填充$k-p-1$的行和列的0元素，这里$p$和卷积的$p$不一样，可以看到这里在进行转置卷积时特征图蓝色部分还是2*2，所以$p$为0；

（3）将卷积核参数上下、左右翻转；

（4）做正常卷积运算

```python
X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
```

```python
tensor([[[[ 0.,  0.,  1.],
          [ 0.,  4.,  6.],
          [ 4., 12.,  9.]]]], grad_fn=<ConvolutionBackward0>)
```

​	

```python
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False) 
tconv.weight.data = K
tconv(X)
```

```python
tensor([[[[4.]]]], grad_fn=<ConvolutionBackward0>)
```





*经过相同超参数卷积再转置卷积后的输出形状不变，但是值会发生变化*



#### 1.19

- 李沐深度学习48全连接卷积神经网络 FCN

全卷积网络（fully convolutional network): 相当于在传统网络的最后的全局平均池化层和全连接层转换为1x1卷积层和转置卷积

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202501192056460.png" alt="image-20250119205608354" style="zoom:50%;" />

使用**双线性插值**的上采样初始化转置卷积层。对于卷积层，我们使用Xavier初始化参数

# 存疑

#### 1.20

- 李沐深度学习49 样式迁移

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202501202149327.png" alt="image-20250120214900194" style="zoom:50%;" />

#### 1.24

- 李沐深度学习51 序列模型 -Ongoing

#### 1.26

- 李沐深度学习51 序列模型 -Ongoing

**自回归**

策略一：**马尔科夫假设**

假设当前数据只跟$\tau$个过去数据点相关

$p(x_t \mid x_1, \ldots, x_{t-1}) = p(x_t \mid x_{t-\tau}, \ldots, x_{t-1}) = p(x_t \mid f(x_{t-\tau}, \ldots, x_{t-1}))$

策略二：**潜变量模型**

引入潜变量$h_t$来表示过去信息$h_t=(x_1,...,x_{t-1})$

$x_t=p(x_t|h_t)$

![image-20250126220420562](https://gitee.com/Goudabao/typora-images/raw/master/202501262204726.png)

#### 1.29

- 李沐深度学习52 文本预处理

#### 2.1

- 李沐深度学习53 语言模型 -Ongoing

**N元语法**当序列很长，文本量不够大，很可能$n(x_1,...,x_T)\leq1$, 使用马尔可夫假设可以缓解这个问题
$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4)\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3)\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3)
\end{aligned}
$$
分别对应一元、二元、三元语法

#### 2.5

- 李沐深度学习53 语言模型

在语言模型任务中，**标签**是下一个词或字符

假设我们有一个句子 "I love machine learning"：

​		输入序列（特征）：`["I", "love", "machine"]`

​		标签（目标）：`["love", "machine", "learning"]`

#### 2.6

- 李沐深度学习54 循环神经网络RNN

**隐变量（Hidden Variable）**

隐藏变量通常指的是在神经网络的隐藏层中存在的变量或节点。它们是网络内部的中间表示，通常不直接可见或可观测，但对最终的输出有重要影响。

**潜变量（Latent Variable）**

潜变量通常指的是在统计模型或生成模型中存在的未观测到的变量。它们是模型中假设存在但未直接观测到的变量，用于解释观测数据的生成过程。

**使用RNN的语言模型**

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202601241745609.png" alt="image-20250206164946232" style="zoom:33%;" />



<img src="https://gitee.com/Goudabao/typora-images/raw/master/202601241745723.png" alt="image-20250206164348762" style="zoom:33%;" />

更新隐藏状态：$\mathbf{h}_t = \phi(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{hx} \mathbf{x}_{t-1} + \mathbf{b}_h)$， 如果去掉$\mathbf{W}_{hh} \mathbf{h}_{t-1}$这一项，就会退化为MLP

输出：$\mathbf{o}_t = \mathbf{W}_{ho} \mathbf{h}_t + \mathbf{b}_o$

# -----------存疑------------

**困惑度**

衡量一个语言模型的好坏可以用平均交叉熵$\pi = \frac{1}{n} \sum\limits_{i=1}^{n} -\log p(x_t \mid x_{t-1}, \ldots)$，$p$是语言模型的预测概率v，$x_t$是真实词

历史原因NLP使用困惑度$exp\pi$来衡量模型，1为完美，无穷大是最差的情况

#### 2.7 

- 李沐深度学习55 循环神经网络 RNN的实现 -Ongoing

- 5分钟搞懂RNN，3D动画深入浅出

![image-20250207135345509](https://gitee.com/Goudabao/typora-images/raw/master/202502071353725.png)

#### 2.8

- 李沐深度学习55 循环神经网络RNN的实现  -Ongoing

**梯度剪裁**（Gradient Clipping）是一种防止梯度爆炸或梯度消失的优化技术，它可以在反向传播过程中对梯度进行缩放或截断，使其保持在一个合理的范围内。

如果梯度长度超过$\theta$，那么投影回长度$\theta$

$\mathbf{g} \leftarrow \min \left( 1, \frac{\theta}{\|\mathbf{g}\|} \right) \mathbf{g}$

#### 2.17

- 李沐深度学习55 循环神经网络RNN的实现  -Ongoing

#### 2.18

- 李沐深度学习55 循环神经网络RNN的实现  -Ongoing

# -------code存疑-------

#### 2.19

- 李沐深度学习55 循环神经网络RNN的实现

#### 2.20

- 李沐深度学习56 门控神经网络GRU

对于一个序列，不是每一个观察值都是同等重要

**更新门Update gate**：关注

**重置门Reset gate**：遗忘

控制这两个门，得到的效果是：当前的输入$X_t$与前一个隐藏层状态$H_{t-1}$分别对这个时间步下隐藏层状态$H_{t}$影响有多大。 

两个极端情况：当前时间步下隐藏层状态$H_{t}$只和当前的输入$X_{t}$相关；当前时间步下隐藏层状态$H_{t}$只和前一个隐藏层状态$H_{t-1}$相关

**门**

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202502201735883.png" alt="image-20250220173502606" style="zoom: 50%;" />

FC layer: Fully Connected layer

**候选隐状态** 

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202502201749795.png" alt="image-20250220174926636" style="zoom:50%;" />

**隐状态**

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202502211752504.png" alt="image-20250221175206321" style="zoom:50%;" />

当$Z_t=1$时，$H_t=H_{t-1}$, 而$Z_{t}=0$时，为RNN

*Q RNN在处理长文本的时候效果不好，那么多长算长？*

*A 尽量避免使用RNN，尝试GRU和LSTM*

#### 2.23

- 李沐深度学习57 长短期记忆网络 LSTM

**忘记门：**将值朝0减少

**输入门：**决定不是忽略掉输入数据

**输出门：**决定是不是使用隐状态



**门**

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202601241745511.png" alt="image-20250223214024715" style="zoom:50%;" />

**候选记忆单元**

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202502232217253.png" alt="image-20250223221736143" style="zoom:50%;" />

**记忆单元**

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202502232220510.png" alt="image-20250223222018394" style="zoom:50%;" />

**隐状态**

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202502232224326.png" alt="image-20250223222446215" style="zoom:50%;" />

C不像H∈[-1， 1]，取值范围更大一些



#### 2.24-4.21

正大杯、华中杯、数学建模大赛、应用统计案例大赛

#### 4.22

- 《》

#### 4.23

- 数据库分类：关系型＆非关系型
- E-R（Entity-Relationship）模型

```mysql
mysql -uroot -p
# 注释
/*注释*/
SHOW database;

USE mysql;

SHOW tables;
```

#### 4.24

**DDL(Data Definition Language)数据定义语言**用于定义和管理数据库的结构，包括库、表、索引、视图等数据库对象的创建、修改和删除，不涉及对数据的操作，而是关注数据库的结构和元数据

```mysql
CREATE DATABASE IF NOT EXISTS 数据库名；

CREATE DATABASE 数据库名 COLLATE 排序规则；

CREATE DATABASE 数据库名 CHARACTER SET 字符集；

CREATE DATABASE 数据库名 CHARACTER SET 字符集 COLLATE 排序规则；
```

#### 4.25

CHARACTER SET 字符集 COLLATE 排序规则

```mysql
CREATE TABLE 表名(
	列名1 INT AUTO_INCREMENT PRIMARY KEY,
	列名2 DATE,
    列名3 VARCHAR(50),
	列名4 VARCHAR(255) NOT NULL,
    列名5 BOOLEAN DEFAULT TRUE
)CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;

SELECT DISTINCT 列名 FROM 表名；

SELECT * FROM 表名 WHERE 条件；
# 数字不需要单引号包裹，文本需要

SELECT * FROM 条件一 AND 条件二；

SELECT * FROM 条件一 OR 条件二；

SELECT * FROM (条件一 OR 条件二) AND 条件三；

SELECT * FROM 表名 ORDER BY 列名 DESC；

SELECT * FROM 表名 ORDER BY 列名 ASC；

SELECT * FROM 表名 ORDER BY 列名1 ASC, 列名2 DESC；

INSERT INTO 表名 VALUES (列的基本信息);

INSERT INTO 表名(列名1, 列名2) VALUES (列1信息, 列2信息);
```

#### 4.26

```mysql
UPDATE 表名 SET 列名 = 新值 WHERE 列名 = 旧值;

UPDATE 表名 SET 列名1 = 新值1, 列名2 = 新值2 WHERE 列名 = 值;

DELETE FROM 表名 WHERE 列名 = 值；# 删除某行

DELETE FROM 表名; # 删除全部行，但表的结构、属性、索引还是完整的

DELETE * FROM 表名; # 同上

# SELECT TOP X * FROM 表名;
# SELECT TOP X PERCENT * FROM; 
```

#### 4.27

```mysql
SELECT * FROM 表名 WHERE 列名 LIKE '%n'; # 查询列名以n结尾的所有行

SELECT * FROM 表名 WHERE 列名 LIKE 'a%o_'; # 查询列名以a开头，倒数第二位为o的所有行

SELECT * FROM 表名 WHERE 列名 IN (值1, 值2, 值3);

SELECT * FROM 表名 WHERE NOT 列名 = 值;

SELECT * FROM 表名 WHERE 列名 != 值;

SELECT * FROM 表名 WHERE 列名 BETWEEN ‘日期1’ AND ‘日期2’;
# BETWEEN AND是闭区间，即左闭右闭

SELECT * FROM 表名 WHERE 列名 IS NULL;

SELECT * FROM 表名 WHERE 列名 IS NOT NULL;

UPDATE customers
SET total_purchases = (
    SELECT SUM(amount)
    FROM orders
    WHERE orders.customer_id = customers.customer_id
)
WHERE customer_type = 'Premium';

SELECT 列名1 FROM 表名1
UNION
SELECT 列名1 FROM 表名2
ORDER BY 列名2;？？？
```

#### 4.28

- 李沐深度学习58 深度循环神经网络

网络加宽会导致overfitting，所以RNN使用多个隐藏层来获取更多的非线性性

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202504281845645.png" alt="image-20250428184459496" style="zoom:50%;" />

#### 4.29

- 李沐深度学习59 双向循环神经网络
- 李沐深度学习60 机器翻译数据集 -Ongoing

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202504291628366.png" alt="image-20250429162843277" style="zoom:50%;" />

双向RNN不能用来预测推理，因为需要提供其前一步和后一步的信息，通常用来对序列抽取特征、填空

#### 4.30

- 李沐深度学习60 机器翻译数据集 -Ongoing

#### 5.4

- 李沐深度学习60 机器翻译数据集

#### 5.5

- 李沐深度学习61 解码器-解码器架构
- 李沐深度学习62 序列到序列学习 -Ongoing

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202601241745667.png" alt="image-20250505171047450" style="zoom:50%;" />

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202505051746999.png" alt="image-20250505174633905" style="zoom:50%;" />

一个模型被分为两个部分：

**编码器**处理输入

**解码器**生成输出

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202505051748912.png" alt="image-20250505174856830" style="zoom:50%;" />

编码器是没有输出的RNN

编码器最后时间步的隐状态用作解码器的初始隐状态

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202505051804534.png" alt="image-20250505180440441" style="zoom:50%;" />

训练时解码器直接使用目标句子（真实值）作为输入，而推理的时候用预测值

![image-20250505181526290](https://gitee.com/Goudabao/typora-images/raw/master/202505051815389.png)

**BLUE**衡量生成序列的好坏
$$
\exp\left(\min\left(0, 1 - \frac{\text{len}_{\text{label}}}{\text{len}_{\text{pred}}}\right)\right) \prod_{n=1}^{k} p_n^{1/2^n}(长匹配有高权重)
$$

其中，$p_n$是预测中所有n-gram的精度。如：标签$A B C D EF$和预测序列$ABBCD$, 有$p_1=\frac{4}{5},p_2=\frac{3}{4},p_3=\frac{1}{3},p_4=0$

#### 5.6

- 李沐深度学习62 序列到序列学习 -Ongoing
- 戴师兄SQL入门-Ongoing

#### 5.7

```mysql
UPDATE customers
SET total_purchases = (
    SELECT SUM(amount) # 选择orders表中amount字段的总和
    FROM orders
    WHERE orders.customer_id = customers.customer_id
)
WHERE customer_type = 'Premium';
# 对于customers表中所有类型为Premium的客户，计算他们在orders表中的所有订单金额总和，并将这个总和设置为这些客户的total_purchases字段的值
```

```mysql
SELECT * FROM employees WHERE last_name LIKE 'smi%' COLLATE utf8mb4_general_ci;
# 选择姓氏以 'smi' 开头的所有员工，不区分大小写
```

- UNION 操作符用于连接两个以上的 SELECT 语句的结果组合到一个结果集合，并去除重复的行。

- UNION 操作符必须由两个或多个 SELECT 语句组成，每个 SELECT 语句的**列数**和对应位置的**数据类型**必须相同
- UNION去除重复行，UNION ALL 不去除重复行

```mysql
SELECT 列名1 FROM 表名1
UNION
SELECT 列名1 FROM 表名2;

SELECT 列名1, 列名2 FROM 表名1
UNION ALL
SELECT 列名3, NULL FROM 表名2
ORDER BY 列名1;

SELECT * FROM 表名 ORDER BY 列名 DESC NULLS FIRST; # 将NULL值排在前面
SELECT * FROM 表名 ORDER BY 列名 DESC NULLS LAST; # 将NULL值排在后面
```

#### 5.10

```mysql
SELECT 列名1, 列名2, 列名3 FROM 表名 ORDER BY 3 DESC, 1 ASC;

SELECT 列名1, aggregate_function(列名2) AS 列名3 
FROM 表名
GROUP BY 列名1;
```

#### 5.11

```mysql
SELECT 列名1, 列名2
FROM 表名1
INNER JOIN 表名2 
ON 表名1.列名 = 表名2.列名;

SELECT o.列名1, 列名2
FROM 表名1 AS o
INNER JOIN 表名2 AS c 
ON o.列名 = c.列名
WHERE 表名X.列名 >='日期';
```

**LEFT JOIN**返回左表的所有行，并包括右表中匹配的行，如果右表中没有匹配的行，将返回 NULL 值

**RIGHT JOIN**同理

```mysql
SELECT 表名1.列名1, 表名1.列名2, 表名2.列名3
FROM 表名1
LEFT JOIN 表名2 
ON 表名1.列名1 = 表名2.列名1;
```

关于**NULL**：

- **IS NULL**当前列的值为NULL，返回TRUE
- **IS NOT NULL**当前列的值不为NULL，返回TRUE
- **<=>**比较操作符（不同于=运算符），当比较两个值相等或者都为NULL时返回TRUE
- 不能使用=NULL或者！=NULL在列中查找NULL值

```mysql
SELECT 列名1 + IFNULL(列名2,0) FROM 表名；
# 列名1和列名2为int型，当列名2中有值为null时，将其转换为0
```

#### 5.12

```mysql
SELECT * FROM 表名1 WHERE 列名1 RLIKE '^[0-9]';
# 以数字开头的所有列

SELECT * FROM 表名1 WHERE 列名1 RLIKE '[^0-9]';
# 包含非数字字符的所有列

ALTER TABLE 表名
ADD COLUMN 列名 DATE;

ALTER TABLE 表名
CHANGE COLUMN 列名 新列名 VARCHAR(255);
# change关键字后紧跟着是要修改的字段名，然后指定新字段名字和类型

ALTER TABLE 表名 MODIFY COLUMN 列名 DECIMAL(10,2);
# DECIMAL用来存储精确数值，包括小数点
# 10是该列可以存储的数字总位数，包括小数点前后的位数;2是小数点后的位数

ALTER TABLE 表名 DROP COLUMN 列名;

ALTER TABLE 表名 ADD PRIMARY KEY (列名);

ALTER TABLE 表名 ADD FOREIGN KEY (列名);

ALTER TABLE 表名 RENAME TO 新表名;

ALTER TABLE 表名 MODIFY 列名 BIGINT NOT NULL DEFAULT 100;
```

#### 主键（Primary Key）

主键是一个或一组字段，用于唯一标识数据库表中的每一行记录。它具有以下特点：

1. **唯一性**：主键字段的值必须唯一，不能有重复的值。
2. **非空性**：主键字段不能接受空值（NULL）。
3. **索引**：MySQL会自动为主键字段创建唯一索引，这有助于提高查询效率。
4. **参照完整性**：主键通常用于在不同的表之间建立关系，作为参照完整性的基础。

#### 外键（Foreign Key）

外键是一个字段或一组字段，它在当前表中引用另一个表的主键。外键用于建立和维护两个表之间的关系，并确保数据的参照完整性。外键具有以下特点：

1. **参照关系**：外键字段的值必须对应于另一个表中的主键字段的值，或者是空值（NULL）。
2. **数据完整性**：外键约束确保了数据的一致性和完整性。如果尝试插入或更新外键字段的值，而该值在参照的主键表中不存在，那么操作将失败。
3. **级联操作**：可以设置外键的级联规则，如`ON DELETE CASCADE`和`ON UPDATE CASCADE`，这意味着当参照的主键记录被删除或更新时，相关的外键记录也会相应地被删除或更新。
4. **索引**：虽然外键字段本身不是索引，但MySQL会在外键字段上创建一个索引，以提高查询效率和确保参照完整性。

#### 主键和外键的关系

- **主键-外键关系**：一个表的主键可以被另一个表的外键引用，从而建立两个表之间的关系。这种关系通常用于实现数据的规范化，减少数据冗余，并确保数据的一致性。
- **一对多关系**：最常见的关系类型是一对多关系，即一个主键可以被多个外键引用。例如，一个客户可以有多个订单，但每个订单只能属于一个客户。在这种情况下，客户表的主键可以被订单表的外键引用。

#### 5.13

| 通配符 | 描述                          |
| ------ | ----------------------------- |
| ^      | 匹配字符串的开始              |
| $      | 匹配字符串的结束              |
| [...]  | 匹配所包含的任意一个字符      |
| [^...] | **匹配未包含的任意字符?????** |
| %      | 匹配任意字符出现任意次数      |
| _      | 匹配任意字符出现一次          |

对于：

| id   | phone\_number |
| ---- | ------------- |
| 1    | 1234567890    |
| 2    | 123-456-7890  |
| 3    | 123456789     |
| 4    | (123) 456-789 |

若输入

```mysql
SELECT * FROM table1 WHERE phone_number RLIKE '[^0-9]';
```

则会输出第二行和第三行的结果，因为它们的phone_number列包含非数字字符

#### 5.14

- Leetcode SQL 50题

#### 5.17

- Leetcode SQL 50题

表： `Weather`

```
+---------------+---------+
| Column Name   | Type    |
+---------------+---------+
| id            | int     |
| recordDate    | date    |
| temperature   | int     |
+---------------+---------+
id 是该表具有唯一值的列。
没有具有相同 recordDate 的不同行。
该表包含特定日期的温度信息
```

编写解决方案，找出与前一天日期相比温度更高的所有日期的 `id` 。

```mysql
SELECT a.id as ID
FROM Weather a
JOIN Weather b ON datediff(a.recordDate, b.recordDate) = 1
WHERE a.Temperature > b.Temperature
```

#### 5.18

- Leetcode SQL 50题

Mysql数据类型：文本TEXT、数字Number、时间/日期Date

TEXT中的**ENUM(x,y,z,etc.)**允许字段的值从一个预定义的值集合中选择

```mysql
CREATE TABLE 表名 (
    字段名 ENUM('值1', '值2', ..., '值N') [NOT NULL | NULL] DEFAULT '默认值'
);
```

例：

```mysql
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    status ENUM('active', 'inactive', 'banned') NOT NULL DEFAULT 'active'
);
```

**IF函数**

```mysql
IF(condition, value_if_true, value_if_false)
```

例

```mysql
SELECT IF(500<1000, "YES", "NO");
```

表: `Activity`

```
+----------------+---------+
| Column Name    | Type    |
+----------------+---------+
| machine_id     | int     |
| process_id     | int     |
| activity_type  | enum    |
| timestamp      | float   |
+----------------+---------+
该表展示了一家工厂网站的用户活动。
(machine_id, process_id, activity_type) 是当前表的主键（具有唯一值的列的组合）。
machine_id 是一台机器的ID号。
process_id 是运行在各机器上的进程ID号。
activity_type 是枚举类型 ('start', 'end')。
timestamp 是浮点类型,代表当前时间(以秒为单位)。
'start' 代表该进程在这台机器上的开始运行时间戳 , 'end' 代表该进程在这台机器上的终止运行时间戳。
同一台机器，同一个进程都有一对开始时间戳和结束时间戳，而且开始时间戳永远在结束时间戳前面。
```

现在有一个工厂网站由几台机器运行，每台机器上运行着 **相同数量的进程** 。编写解决方案，计算每台机器各自完成一个进程任务的平均耗时。

完成一个进程任务的时间指进程的`'end' 时间戳` 减去 `'start' 时间戳`。平均耗时通过计算每台机器上所有进程任务的总耗费时间除以机器上的总进程数量获得。

结果表必须包含`machine_id（机器ID）` 和对应的 **average time（平均耗时）**别名 `processing_time`，且**四舍五入保留3位小数。**

```mysql
# 解法1：
SELECT
    machine_id,
    round(2*avg(if(activity_type = 'start',-1,1)*timestamp),3) as processing_time
FROM 
    Activity
GROUP BY
    machine_id
    
# 解法2：
SELECT
    machine_id,
    ROUND(AVG(CASE WHEN activity_type='end' THEN timestamp END)-AVG(CASE WHEN activity_type='start' THEN timestamp END),3) AS processing_time
FROM
    Activity
GROUP BY
    machine_id;
    
# 解法3：
SELECT machine_id AS 'machine_id',
    ROUND(
        SUM(IF(activity_type = 'start', -timestamp, timestamp))
        / COUNT(DISTINCT process_id) 
        ,3) AS 'processing_time'
FROM Activity
GROUP BY machine_id;
```

#### 5.19

- Leetcode SQL 50题

**CROSS JOIN 交叉连接**返回被连接两个的笛卡尔积，结果的行数等于两个表行数的乘积（后加条件只能是WHERE, 不能加ON）

```mysql
SELECT * 
FROM 表1
CROSS JOIN 表2 
```

#### 5.20

- Leetcode SQL 50题

#### 5.21

- Leetcode SQL 50题

**COUNT**函数：计算表中所有行的数量；计算特定行中非NULL值的数量（*0也会被计数*）

```mysql
SELECT COUNT(*) FROM 表名;

SELECT COUNT(列名) FROM 表名;

SELECT 列名1, COUNT(*) FROM 表名 GROUP BY 列名1;
# 根据列名1进行分组，并返回每个分组的行数
```

**WHERE子句**

`WHERE`子句用于在数据分组之前对数据进行过滤。它通常用于对原始数据进行条件筛选，即在数据被聚合或分组之前就进行筛选。`WHERE`子句可以对表中的任何列进行条件筛选，包括那些没有参与聚合函数的列

```mysql
SELECT region, product, SUM(amount) AS total_sales
FROM sales
WHERE amount > 10000
GROUP BY region, product;
```

**HAVING子句**

`HAVING`子句用于在数据分组之后对数据进行过滤。它通常用于对聚合结果进行条件筛选，即在数据已经被聚合或分组之后进行筛选。`HAVING`子句可以对聚合函数的结果进行条件筛选，如`SUM()`、`COUNT()`、`AVG()`等。

```mysql
SELECT region, SUM(amount) AS total_sales
FROM sales
GROUP BY region
HAVING SUM(amount) > 10000;
```

#### 5.27

- 多窗口效果：`视图`--`新建窗口`

EXCEL的日期对应数字：1→1900年1月1日

- 绝对引用：位置坐标前加`$`  (快捷键` F4`）
- ***SUMIF***(判断条件所在区域, 条件, 求和区域)
- ***IFERROR***(VALUE, 报错时返回的VALUE)
- ***SUMIFS***(求和区域, 条件1所在区域, 条件1表达式, 条件2所在区域, 条件2表达式)

#### 6.6

#### Excel 函数

- 快速求和：`ALT` +` =`

- ***LEFT***(文本字符串, 提取字符的数量) # 默认取1个
- ***MID***(文本字符串, 要提取字符的起点, 提取字符长度)

文本数据*1可以将其转换为数值数据

数值数据&“”可以将数值数据转为文本数据

**文本**数据在单元格中**靠左**

**数字**以及**日期**数据在单元格中**靠右**

- ***VLOOKUP***(匹配依据, 查找区域, 查找数据在第几列, 0/FALSE精确匹配1/TRUE近似匹配) 

*查找区域第一列必须为匹配列；尽量使用精确匹配（0或者FALSE）*

双击单元格下边框→移至数据最底部

双击单元格右下角→批量复制单元格格式

#### 6.10

- 李沐深度学习62 序列到序列学习 -Ongoing

**Embedding**嵌入层：将离散数据映射为连续变量，将文本转换为连续向量

通俗解释见https://zhuanlan.zhihu.com/p/164502624

#### 6.11

#### Excel Power Query

没有撤销操作，只能删掉历史步骤

左连接/右连接/内连接/外连接

#### 6.12

#### 数据透视表

#### 6.13

#### 数据透视表

- ***GETPIVOTDATA***(数据透视表字段名，数据透视表位置，字段名1，字段需要满足的查询条件1，字段名2，字段需要满足的查询条件2, ...)

被认为是透视表中的VLOOKUP函数

如：=GETPIVOTDATA("成交金额",数据2!F3,"业务组",一、基础题!C42,"期数",一、基础题!D42)

#### 6.17

对于这样一组数据：

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202506171249982.png" alt="image-20250617124936832" style="zoom: 50%;" />

**跨越合并：**逐行合并，每行合并为一个，即有几行就合并为几个单元格

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202506171250196.png" alt="image-20250617125057138" style="zoom:50%;" />

**合并：**所选单元格合并为一个单元格

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202506171251592.png" alt="image-20250617125116530" style="zoom:50%;" />

**0!.0,"w"**将数值转换为以w为单位

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202506172327821.png" alt="image-20250617232724729" style="zoom: 50%;" />

#### 6.18

**[颜色10]0%▲;[红色]-0%▼**将数值显示为:

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202506181904103.png" alt="image-20250618190437004" style="zoom:50%;" />

#### 仪表盘制作

![image-20250618191525728](https://gitee.com/Goudabao/typora-images/raw/master/202506181915808.png)

#### 6.21

- Tableau

#### 6.22

**ABtest**是一种统计方法，用于比较两个或多个版本的产品、服务或营销策略的效果，目的是确定哪个版本在特定目标上表现更好。以下是A/B测试的简要概述：

1. **目标设定**：明确测试的目标，如提高用户参与度、增加转化率或提升用户满意度。
2. **版本设计**：创建两个或多个版本，通常一个作为控制组（A），其他作为实验组（B、C等），每个版本在某个特定变量上有所不同。
3. **随机分配**：将用户随机分配到不同的版本组，确保每个用户只能看到其中一个版本。
4. **数据收集**：在测试期间收集用户的行为数据，如点击、购买、注册等。
5. **统计分析**：使用统计方法分析收集到的数据，确定不同版本的效果差异是否具有统计显著性。
6. **结果评估**：根据分析结果评估哪个版本更有效，并决定是否将其作为标准版本。
7. **决策实施**：将表现最佳的版本推广到所有用户，以优化整体性能。

#### 6.23

- Tableau

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202506232229967.png" alt="image-20250623222907716" style="zoom: 33%;" />

<img src="https://gitee.com/Goudabao/typora-images/raw/master/202506232230074.png" alt="image-20250623223001956" style="zoom:33%;" />

#### 6.24

- Tableau

![screenshots](https://gitee.com/Goudabao/typora-images/raw/master/202506241740669.gif)

[Tableau仪表盘演示](https://public.tableau.com/app/profile/.52123632/viz/test_17506863675130/1?publish=yes)

#### 6.27

- SQL

limit x,y 取第x行后面y行数据

```mysql
SELECT * 
FROM 表名
LIMIT 5,5
# 取出表中第6-10行数据
```

JOIN默认为内连接

#### 6.29

- SQL

#### 7.1

**信息熵**是对数据集中不确定性或混乱程度的度量。在一个数据集中，如果各类别的样本数量大致相等，那么该数据集的熵值较高，表示不确定性较大；反之，如果某个类别的样本数量占据绝对优势，那么熵值较低，表示不确定性较小。因此，信息熵可以用来评估数据集的纯度。
设$X$是一个有限值的离散随机变量，其概率分布为：
$$
P(X=x_i)=p_i,i=1,2,...n
$$

假设变量$x$的随机取值为$X = x_1,x_2,x_3,...,x_n$，每一种取值的概率分布为$p_1,p_2,...,p_i$,则变量$X$的熵为：
$$
H(X)=-\sum_{i=1}^n{p_ilog_{2}{p_i}}
$$
**条件熵**是在给定某个随机变量取值的情况下，另一个随机变量的不确定性度量。在决策树中，条件熵用于评估在给定某个特征条件下数据集的纯度。通过比较划分前后数据集的条件熵变化，我们可以选择出能够最大程度降低不确定性的划分属性。

在已知某一随机变量$Y$的条件下，另一随机变量$X$的不确定性。其公式为：
$$
H(X|Y) = \sum P(y) \cdot H(X|y) = \sum P(y) \cdot \left[ -\sum P(x|y) \cdot \log_2 P(x|y) \right]
$$

#### 7.3

**辛普森悖论**是一种统计现象，指的是在不同组数据中都显示出某种趋势，但当这些数据汇总在一起时，原本的趋势却消失了，甚至可能出现相反的趋势。

**解决办法**

1. 分层分析

2. 控制混淆变量
3. 可视化与统计检验
4. 建立因果模型

辛普森悖论无法完全避免的，很多问题，完全依靠统计学推导因果关系无法实现。就拿生产环境数据来说，虽然我们做了各种画像，但是其他分类方式依然存在，理论上的潜在变量会无穷无尽。

我们能做的，就是仔细认真的研究各种影响因素，不要笼统概括的看问题，尤其数据分析问题，拆解的越细，最终得到的效果越好。

关于避免辛普森悖论的出现，目前比较流行的一种做法，就是需要斟酌个别分组的权重，以一定的系数去消除以分组资料基数差异所造成的影响，同时必须了解该情境是否存在其他潜在因素，需要进行综合性考虑。

这段话看完有点晕圈，在实际中斟酌权重和判断其他因素，大多数还是更多依赖经验。

虽然不能根本上避免辛普森悖论，但我们至少应该明白：在因果关系里，量与质是不等价的，但是量比质更容易测量，所以人们总是习惯用量来评定好坏，而该数据却不是重要的。

#### 7.4

**用户留存率**某一天新增的用户在之后的第N天仍然登录的比例，成为第N天的留存率

1. **次日留存率** = 当日活跃用户数 / 前一日总用户数 × 100%
2. **次周留存率** = 当周活跃用户数 / 前一周总用户数 × 100%
3. **次月留存率** = 当月活跃用户数 / 前一个月总用户数 × 100%

#### 7.5

- SQL

**多表连接计算留存率**

```mysql
SELECT DATE(tu.register_time),
		100*count(DISTINCT t1.uid)/count(DISTINCT tu.id) rr1,
		100*count(DISTINCT t2.uid)/count(DISTINCT tu.id) rr3,
		100*count(DISTINCT t3.uid)/count(DISTINCT tu.id) rr7,
		100*count(DISTINCT t4.uid)/count(DISTINCT tu.id) rr30,
FROM t_user tu
LEFT JOIN t_user_login t1 ON (t1.uid = tu.id AND DATE(t1.login_time) = DATE(tu.register) + INTERVAL '1' DAY)
LEFT JOIN t_user_login t2 ON (t2.uid = tu.id AND DATE(t2.login_time) = DATE(tu.register) + INTERVAL '3' DAY)
LEFT JOIN t_user_login t3 ON (t3.uid = tu.id AND DATE(t3.login_time) = DATE(tu.register) + INTERVAL '7' DAY)
LEFT JOIN t_user_login t4 ON (t4.uid = tu.id AND DATE(t4.login_time) = DATE(tu.register) + INTERVAL '30' DAY)
GROUP BY DATE(tu.register_time);
# 不推荐，因为效率比较低下
```



#### 7.6

- SQL

```mysql
SELECT *
FROM t_user tu
LEFT JOIN t_user_login tul
ON (t1.uid = tu.id 
AND (DATE(tul.login_time) = DATE(tu.register_time) + INTERVAL '1' DAY
    OR DATE(tul.login_time) = DATE(tu.register_time) + INTERVAL '3' DAY
    OR DATE(tul.login_time) = DATE(tu.register_time) + INTERVAL '7' DAY
    OR DATE(tul.login_time) = DATE(tu.register_time) + INTERVAL '30' DAY))
```

#### 7.7

- SQL

#### 7.8

- SQL

**窗口函数**是SQL中一种特殊类型的函数，它允许你对一组行执行计算，这些行与当前行有某种关系，比如它们可能是查询结果集中当前行的一部分。窗口函数不会像聚合函数那样合并行，而是保留原始数据集中的每一行，为每行生成一个结果值。

窗口函数通常与 `OVER()` 子句一起使用，该子句定义了窗口函数操作的“窗口”。窗口可以是查询结果集中的一个子集，这个子集可以是行的一个序列，这些行与当前行有某种关系，比如它们可能是当前行的前面几行、后面几行，或者是当前行及其前后的行。

1. **不需要分组**： 窗口函数不需要对数据进行分组，它可以在不改变原始数据集结构的情况下对数据进行计算。
2. **保留原始数据**： 与聚合函数不同，窗口函数不会合并行，而是为每一行计算一个值。
3. **支持复杂的计算**： 窗口函数可以执行复杂的计算，如累积求和、移动平均、排名等。

常见的窗口函数有：

1. **`ROW_NUMBER()`**： 为结果集中的每一行分配一个唯一的连续整数，通常用于数据的分页。
2. **`RANK()`**： 为结果集中的每一行分配一个排名，相同值的行会分配相同的排名，并在后续的排名中留下空位。
3. **`DENSE_RANK()`**： 与 `RANK()` 类似，但是相同值的行会分配相同的排名，并且后续的排名不会留下空位。
4. **`NTILE(n)`**： 将结果集的行分配到指定数量的组（或“桶”）中，每个组大约包含相同数量的行。
5. **`LEAD(column)` 和 `LAG(column)`**： 分别获取当前行的下一行或上一行的指定列的值。
6. **`SUM()`、`AVG()`、`MIN()`、`MAX()`**： 这些聚合函数也可以作为窗口函数使用，对一个窗口内的行进行计算。

```mysql
SELECT 
    salesperson,
    total_sales,
    RANK() OVER (ORDER BY total_sales DESC) AS sales_rank
FROM 
    sales;
 #查询会返回每个销售员的姓名、总销售额以及他们在总销售额上的排名
```

**WITH子句**（也称为**公用表表达式**或**CTE**，Common Table Expression）允许你定义一个或多个临时的结果集，这些结果集可以在主查询中被引用。`WITH` 子句通常用于简化复杂的查询，使得查询更加清晰和易于维护。

```mysql
WITH cte_name (column1, column2, ...)
AS
(
    -- CTE definition (a subquery)
    SELECT column1, column2, ...
    FROM ...
    WHERE ...
)
SELECT *
FROM cte_name;
```



```mysql
WITH total_sales AS (
    SELECT region, product, SUM(amount) AS total_amount
    FROM sales
    GROUP BY region, product
)
SELECT region, product, total_amount
FROM total_sales
WHERE total_amount > 10000;
```

**CASE WHEN 语句**语句是一种条件表达式，它允许你在查询结果中基于不同的条件来返回不同的值。这种表达式在处理复杂的逻辑时非常有用，因为它允许你在单个表达式中执行类似于编程语言中的 `if-else` 语句的操作

```mysql
CASE 
    WHEN condition1 THEN result1
    WHEN condition2 THEN result2
    ...
    ELSE resultN
END
```

```mysql
SELECT 
    employee_id,
    name,
    salary,
    department,
    CASE 
        WHEN department = 'Sales' THEN salary * 0.1
        ELSE salary * 0.05
    END AS bonus
FROM 
    employees;
```

在`UPDATE`语句中使用

```mysql
UPDATE employees
SET salary = CASE 
                   WHEN performance = 'Good' THEN salary * 1.10
                   WHEN performance = 'Average' THEN salary * 1.05
                   ELSE salary
               END
WHERE employee_id = 123;
```

在`WHERE`语句中使用

```mysql
SELECT *
FROM employees
WHERE salary > CASE 
                 WHEN department = 'Sales' THEN 50000
                 ELSE 30000
               END;
```

**窗口函数计算留存率**

```mysql
WITH t1 AS (
SELECT u.id, u.user_name, date(u.register_time) reg_date, date(l.login_time) login_date,
       DENSE_RANK() OVER (PARTITION BY date(u.register_time) ORDER BY u.id) daily_reg,
       DENSE_RANK() OVER (PARTITION BY date(u.register_time), date(l.login_time) ORDER BY l.uid) daily_login
FROM t_user u
LEFT JOIN t_user_login l 
ON (l.uid = u.id AND date(l.login_time) BETWEEN date(u.register_time) + INTERVAL '1' DAY AND date(u.register_time) + INTERVAL '30' DAY)
),
t2 AS (
SELECT reg_date, max(daily_reg) daily_reg, login_date, max(daily_login) daily_login
FROM t1
GROUP BY reg_date, login_date)
SELECT reg_date, max(daily_reg),
       100*max(CASE WHEN login_date = reg_date + INTERVAL '1' DAY THEN daily_login END)/max(daily_reg) rr1,
       100*max(CASE WHEN login_date = reg_date + INTERVAL '3' DAY THEN daily_login END)/max(daily_reg) rr3,
       100*max(CASE WHEN login_date = reg_date + INTERVAL '7' DAY THEN daily_login END)/max(daily_reg) rr7,
       100*max(CASE WHEN login_date = reg_date + INTERVAL '30' DAY THEN daily_login END)/max(daily_reg) rr30
FROM t2 
GROUP BY reg_date;
```

#### 7.10

- SQL

[在线SQL](https://sqlfiddle.com/sql-server/online-compiler)

#### 7.11

- PowerBI

![screenshots](https://gitee.com/Goudabao/typora-images/raw/master/202507120151087.gif)

#### 7.14

- 杜恩泽Power Query

![image-20250714223503869](https://gitee.com/Goudabao/typora-images/raw/master/202507142235134.png)

#### 7.15

- PQ处理异常值、清洗数据、合并数据

#### 7.16

- PQ追加查询（合并）、连接MySQL

#### 7.17

- Excel练习题1—基础



- ***SORT***(源数据, 排序依据, 升序/降序, 按照列/行)
- ***XLOOKUP***(找什么，在哪儿找，返回哪儿，[找不到给啥]，[匹配方式]，[搜索模式])
- 多条件***XLOOKUP***(1, (条件1)*(条件2), 结果列)

#### 7.18

- 杜恩泽Excel

#### 7.19

- Excel练习题1—综合

#### 7.20

- Excel练习题2—基础

- ***Find***（找什么，在哪儿找，[从第几位开始找]）寻找某文本在字符串中第几位出现（区分大小写）

XLOOKUP和VLOOKUP只会返回一个匹配值，XLOOKUP可以通过最后一个参数[1 / -1]选择正序还是倒序查找

#### 7.21

- Excel练习题2—综合-Ongoing

#### 7.22

- Excel练习题2—综合

#### 7.23

#### **Q数据预处理的方法以及工具**

#### **A**

预处理：缺失值、异常值、数据转换、特征工程、数据规范化、编码分类器

工具：Excel、Python、R、SQL、Tableau、Power BI

#### **Q指标体系的构建**

#### **A**

##### 第一步：明确目标

首先得清楚你要解决的问题是啥，或者你要达成的目标是啥。比如，你要是想提升网站用户活跃度，那你的指标体系就得围绕用户行为来设计。

##### 第二步：了解业务

搞清楚你的业务是咋回事，业务流程是啥样的。这样你才能知道哪些环节是关键的，哪些数据是重要的。

##### 第三步：定义指标

根据目标和业务流程，定义出一系列指标。指标得是可量化的，比如用户活跃度，你可以通过日活跃用户数（DAU）来衡量。

##### 第四步：分类指标

指标多了，得分类管理。常见的分类方法有：

- **结果类指标**：直接反映业务成果的，比如销售额、利润。
- **过程类指标**：反映业务过程中的关键点，比如用户留存率、转化率。
- **质量类指标**：衡量业务质量的，比如客户满意度、产品缺陷率。

##### 第五步：构建体系

把指标按照逻辑关系组织起来，形成一个体系。比如，一个大指标下面可以有多个小指标，小指标支撑大指标。

##### 第六步：数据收集

确定好指标后，得有数据支撑。你得知道这些指标的数据从哪儿来，怎么收集。

##### 第七步：分析和优化

用这些指标去分析业务，看看哪儿做得好，哪儿需要改进。然后根据分析结果不断调整和优化你的指标体系。

#### **Q设计问卷的角度**

#### **A**

##### 1. 明确目的

首先得清楚你为啥要设计这个问卷，你想了解啥信息。这就像是写信前得知道你想跟朋友说啥。

##### 2. 了解受众

你得知道你的问卷是给谁填的，他们的背景、兴趣、习惯是啥。这样你才能设计出他们愿意回答的问题。

##### 3. 问题要清晰

问题得写得明明白白的，别让人家看不懂你在问啥。用简单直白的话，别整那些复杂的术语。

##### 4. 选项要全面

如果是选择题，选项得全面，别让用户想找的答案找不到。这就跟你给朋友推荐餐厅，得确保他们喜欢的菜系你都有推荐。

##### 5. 问题别太多

问卷太长了没人愿意填。挑重点问题问，别啥鸡毛蒜皮的事儿都问。

##### 6. 保护隐私

如果问到个人信息，得告诉用户你为啥要这些信息，怎么保护他们的隐私。

##### 7. 逻辑顺序

问题得按逻辑顺序来，别让用户感觉东一榔头西一棒子的。就像讲故事，得有开头、中间、结尾。

##### 8. 测试问卷

设计好后，找几个人来试试填，看看他们有没有哪里不懂或者不舒服的地方。

##### 9. 易于分析

你还得考虑最后怎么分析这些数据，所以问题和选项得设计得容易统计。

##### 10. 感谢参与者

最后，别忘了在问卷末尾感谢人家花时间帮你填问卷。

#### Q如何筛选掉无用的问卷

#### A

##### 1. 设定筛选标准

首先，你得知道什么样的问卷算是无效的。比如，问卷没填完、填得飞快、答案互相矛盾，或者明显是乱填的。

##### 2. 检查填写时间

如果问卷设计得很合理，一般人需要5分钟来完成，结果有人30秒就填完了，那很可能就是乱填的。

##### 3. 逻辑一致性检查

问卷里的问题得有逻辑，比如年龄不能是负数，性别不能既选男又选女。如果发现逻辑不一致，那问卷可能就无效。

##### 4. 答案模式识别

有些人可能为了省事，就选同一个选项到底。如果发现答案模式太规律，也可能是无效问卷。

##### 5. 开放性问题检查

如果问卷里有开放性问题，可以看看回答是否认真。如果回答太简单或者明显是敷衍，那问卷的有效性就值得怀疑。

##### 6. 使用软件工具

现在有很多问卷分析软件，它们能帮你自动识别一些无效问卷，比如通过分析填写时间和答案模式。

##### 7. 设定筛选规则

在问卷收集平台（比如问卷星、Google 表单等）上，你可以设置一些自动筛选规则，比如“未完成问卷自动排除”。

##### 8. 人工审核

有时候机器筛选不够准确，还需要人工再检查一遍。可以找几个人一起看，确保筛选的准确性。

##### 9. 反馈机制

可以在问卷末尾加一个问题，比如“您是否认真填写了问卷？”这样也能帮你识别一些无效问卷。

##### 10. 透明度

告诉参与者你会筛选问卷，这样他们可能会更认真地填写。

面试的时候，你可以结合自己的经验，说说你是怎么设定筛选标准的，用了哪些工具，或者你是怎么通过人工审核来确保问卷质量的。这样能展示你的细心和专业性。

#### 9.1

***mod（a,b)***指的是a/b的余数

```mysql
mod(id,2) = 1 # id为奇数

mod(id,2) = 0 # id为偶数
```

#### 9.10

找课题

#### 9.11

找课题

#### 9.18

找课题

| 文件类型     | 优点                                           | 缺点                                                         | 适用场景                                                     |
| ------------ | ---------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **JSON**| 人类可读，结构灵活，跨语言通用                 | 对于纯数字数据，体积较大，读写效率不如二进制                 |**配置文件**、**API通信**、**存储元数据和标注**              |
| **NPY**| 读写极快，节省空间，完美保留数据类型和数组结构 | 二进制格式，不方便直接查看和编辑                             |**存储机器学习模型权重**、**处理好的训练/测试数据**、任何大型数值数组 |
| **PNG/JPG**| 压缩率高，文件小，通用性好                     | 有损压缩（JPG），读写需要编解码，不适合存储原始计算数据      |**最终结果的可视化**、网页图片、日常照片存储                 |
| **CSV/XLSX**| 直观，易于用Excel等软件编辑                    | 只能表示二维表格，处理大型数据时效率低，文本存储数字浪费空间 |**存储简单的表格数据**、数据分析的初始阶段、与非技术人员共享数据 |

#### 9.24

- SQL

#### 10.2

- SQL

**笛卡尔积**是一个数学概念，在数据库领域，它指代的是将两个表的每一行与另一个表的每一行进行组合，从而得到所有可能的组合结果。

所有 `JOIN` (如 `INNER JOIN`, `LEFT JOIN`) 的逻辑基础都可以看作是先进行笛卡尔积，然后再根据 `ON` 条件进行筛选。虽然数据库优化器不会真的去生成这个庞大的中间表，但从逻辑上理解这一点至关重要。

- ***ROUND***(number, [decimals]）用于将一个数值四舍五入到指定的小数位数, 默认不保留

`GROUP BY` 会打乱输出顺序

**使用` AVG(IF(...)) ` 是计算比率的更简洁和安全的方法，它能正确处理所有情况，包括某用户没有任何确认记录的情况**

```mysql
SUM(IF(action = 'confirmed', 1, 0)) / COUNT(action)

AVG(action = 'confirmed')
```

#### 10.3

- SQL

#### 10.6

- SQL

**在进行中间计算时**：保留 `AVG()` 的完整精度。

**在 `SELECT` 子句中准备最终输出时**：使用 `ROUND()` 来格式化结果，使其满足可读性和业务需求

#### 10.7

- SQL

#### 10.8

- SQL

##### SQL 查询的逻辑执行顺序

1. **`FROM`**
   - 这是所有查询的起点。它指定了数据来源的表。
   - 如果包含 `JOIN`，此时会根据 `ON` 条件将多个表联接起来，形成一个庞大的虚拟中间表（可以理解为笛卡尔积之后用 `ON` 条件筛选）。
2. **`WHERE`**
   - 在 `FROM` 步骤生成的虚拟表上，应用 `WHERE` 子句中的过滤条件。
   - 只有满足 `WHERE` 条件的行才会被保留下来，进入下一步。
   - **注意**：此时还不能使用 `SELECT` 中定义的别名，因为 `SELECT` 还没有执行。
3. **`GROUP BY`**
   - 将经过 `WHERE` 筛选后的结果行，按照 `GROUP BY` 子句中指定的列进行分组。
   - 所有在指定列上具有相同值的行会被分到同一个组中。
4. **`HAVING`**
   - 对 `GROUP BY` 之后形成的分组进行过滤。
   - 只有满足 `HAVING` 条件的**分组**才会被保留下来。
   - `WHERE` 过滤的是**行**，而 `HAVING` 过滤的是**分组**。`HAVING` 子句中可以使用聚合函数（如 `COUNT()`, `SUM()`, `AVG()`），而 `WHERE` 不可以。
5. **`SELECT`**
   - 到了这一步，才开始处理 `SELECT` 子句。
   - 它会选择要最终显示的列或表达式。
   - 如果 `SELECT` 中有 `DISTINCT` 关键字，会在此时对结果进行去重。
   - 此时，为列定义的别名（如 `AVG(price) AS average_price`）才生成。
6. **`ORDER BY`**
   - 对 `SELECT` 步骤产生的最终结果集进行排序。
   - 这是少数几个可以使用 `SELECT` 中定义的列别名的地方。
7. **`LIMIT` / `OFFSET`**
   - 在结果集排序完成后，`LIMIT` 和 `OFFSET` 会从最终结果中选取指定的行。
   - `LIMIT` 指定返回的最大行数，`OFFSET` 指定从哪里开始返回。这是最后执行的步骤。

可以通过一个简单的口诀来记忆这个顺序：

**F**rom**W**here**G**roupers**H**ave**S**elected**O**rdered**L**imits. (来自哪里(Where)的石斑鱼(Groupers)已经(Have)选择(Selected)了有序的(Ordered)限制(Limits))

对应关系：

- **F**rom -> `FROM` / `JOIN`
- **W**here -> `WHERE`
- **G**roupers -> `GROUP BY`
- **H**ave -> `HAVING`
- **S**elected -> `SELECT`
- **O**rdered -> `ORDER BY`
- **L**imits -> `LIMIT`

#### 10.16

- SQL

***DATE_FORMAT***(date, format)将日期更改为指定格式

```mysql
SELECT DATE_FORMAT(order_time, '%Y-%m-%d') FROM orders;
-- 结果: '2023-10-26'

SELECT DATE_FORMAT(order_time, '%Y年%m月%d日 %H点%i分') FROM orders;
-- 结果: '2023年10月26日 17点08分'

SELECT DATE_FORMAT(order_time, '%Y-%m') AS year_month FROM orders;
-- 结果: '2023-10'

SELECT DATE_FORMAT(order_time, '%M %d, %Y at %h:%i %p') FROM orders;
-- 结果: 'October 26, 2023 at 05:08 PM'
```

**`GROUP BY` 是为了“聚合”**：它将多行数据**压缩**成一行，用于生成汇总报告（比如计算每个部门的总工资）。

**`PARTITION BY` 是为了“开窗”**：它不改变原始数据的行数，而是为每一行数据**添加**一个基于其所在分组的计算结果（比如计算每个员工的工资占其部门总工资的百分比）。

| 特性             | `GROUP BY`                            | `PARTITION BY`                                               |
| ---------------- | ------------------------------------- | ------------------------------------------------------------ |
| **主要目的**|**聚合 (Aggregation)**|**分区 (Partitioning)**，为窗口函数做准备                    |
| **对行数的影响**|**减少行数**，每个分组最终只返回一行  |**不改变行数**，为原始的每一行都添加一个计算列               |
| **返回结果**| 返回每个分组的**摘要行**| 返回**原始的所有行**，并附带一个额外的计算列                 |
| **典型函数**     | 聚合函数: `SUM()`, `COUNT()`, `AVG()` | 窗口函数: `ROW_NUMBER()`, `RANK()`, `LEAD()`, `SUM() OVER()` |
| **使用场景**     | 生成汇总报表，如“每个部门的平均工资”  | 计算排名、累计总和、行间比较，如“每个部门内工资最高的员工”   |

#### 10.20

- 找课题

#### 12.5

- Power BI

Power Query 使用的是 **M 语言**

| **特性**|**M 语言**|**DAX 语言**              |
| ------------ | -------------------- | ------------------------- |
| **作用阶段**|**提取和转换 (ETL)**|**建模和分析 (BI)**       |
| **输出结果**| 干净的**表**|**度量值**或**新列**    |
| **上下文**   | 步骤和查询           | 行和筛选器                |
| **大小写**|**区分**(严格)      |**不区分** (推荐大写函数) |

#### 12.9

不同模型之间的好坏区别

优化算法

#### 12.10

```python
!pip install tabulate
```

 **！(感叹号)**

**含义：** “切换到系统命令行模式”。

**作用：** 相当于你在 Windows 的 CMD 窗口或者 Mac/Linux 的 Terminal 终端里敲命令。

#### 2026.1.24

#### 个人知识库搭建

- **项目初始化**：完成本地 Git 仓库搭建，配置 `.gitignore` 排除 `node_modules` 等冗余文件。
- **CI/CD 部署**：基于 Vercel + GitHub 实现自动化构建，解决 `base` 路径导致的 404 问题，修正 Output Directory 为 `.vitepress/dist`。
- **内容架构**：
    - 定制首页 (`index.md`)：配置 Hero 区域与快速入口。
    - 配置侧边栏 (`config.mts`)：实现文档自动导航与分类。
- **流程自动化**：编写 Windows 批处理脚本 (`.bat`)，集成 `add/commit/push` 及自动时间戳功能，解决 CMD 路径跳转与编码乱码问题。
- **网络优化调研**：确定 Spaceship `.top` 域名方案以解决国内访问问题。太贵了，故Pass。

[Howard的主页](https://my-notes-pearl-three.vercel.app/)

#### 1.25

接单￥350

**Q1 (回归分析)**

- **做了什么**：检验动量因子有没有钱赚。
- **结论**：**动量崩了**（系数是负数 -3.58），截距显著。

**Q2 (投资组合 VBA)**

- **做了什么**：写个宏，自动算 AMZN 和 JPM 组合的收益。
- **结果**：**预期收益 1.57%，风险 6.22%**。
- **重点**：代码用 `Sub`，要手动点运行。

**Q3 (蒙特卡洛模拟)**

- **做了什么**：模拟未来 50 种股价走势，算回望期权价格。
- **结果**：价格在 **11.91** 左右。
- **重点**：Data Table 表头必须填公式 `=B30`。

**Q4 (二叉树定价)**

- **做了什么**：写两个函数算“牛市价差”策略的价格。
- **结果**：价格是 **11.73**。
- **重点**：代码用 `Function`，必须放在“模块”里。