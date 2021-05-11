# Loss

## GIoU(Generalized IoU)

定义：

1.$IoU=\frac{A\cap B}{A\cup B}$

2.$GIoU=IoU -\frac{|\backslash (A\cap B)|}{|C|}$

3.$L_{IoU}=1-IoU$

4,$L_{GIoU}=1-GIoU$

#### 优点/改进：

反映了两个框的距离

#### ref

https://blog.csdn.net/weixin_41735859/article/details/89288493

## Smooth L1 Loss

````math

Smooth L_1=\begin{cases}
0.5x^2,|x|<1,\\
|x|-0.5,otherwise
\end{cases}
````

#### 优点：回避了L1和L2的缺点

从损失函数对x的导数可知： ![[公式]](https://www.zhihu.com/equation?tex=L_%7B1%7D)损失函数对x的导数为常数，在训练后期，x很小时，如果learning rate 不变，损失函数会在稳定值附近波动，很难收敛到更高的精度。 ![[公式]](https://www.zhihu.com/equation?tex=L_%7B2%7D) 损失函数对x的导数在x值很大时，其导数也非常大，在训练初期不稳定。![[公式]](https://www.zhihu.com/equation?tex=smooth_%7BL_%7B1%7D%7D%5Cleft%28+x+%5Cright%29+)** 完美的避开了 **![[公式]](https://www.zhihu.com/equation?tex=L_%7B1%7D)**和 **![[公式]](https://www.zhihu.com/equation?tex=L_%7B2%7D)**损失的缺点。**

#### 缺点：与IoU不等价

#### ref

https://zhuanlan.zhihu.com/p/104236411

## DIoU(Distance-IoU Loss)

考虑了**重叠面积**和**中心点距离**

IoU-based loss:

$L=1-IoU+R(B,B^{gt})$，其中$R(B,B^{gt})$代表惩罚项

DIoU：

$L_{DIoU}=1-IoU+\frac{\rho^2(b,b^{gt})}{c^2}$，其中$\rho$为欧氏距离，$c$为C（最小包括框）的对角线长度

#### 优点

比GIoU收敛快

## CIoU(Complete IoU)

$R_{CIoU}=\frac{\rho^2(b,b^{gt})}{c^2}+\alpha\nu$

#### ref

https://zhuanlan.zhihu.com/p/104236411

## Focal Loss

## Cross Entropy Loss

* Self-information:$I(X)=-log(P(X))$$

* Entropy: Expectation of self-information:$H(X)=E_{X \sim P}[I(X)]$

* KL Divergence: $KL(P|Q)=D_{KL}(P|Q)=\sum_{x\in X}P(x)(logP(x)-log(Q(X))$

````
expectation of the log difference between the probability of data in the original distribution with the approximating distribution. 

P=>real distribution Q=>our distribution
````

* Cross-Entropy: $HH_P(Q)=-E_{X\sim P}log(Q(X))=-\sum_{x\in X}P(x)log(Q(x))$

#### relationship

$H(P|Q)-H(P)=D_{KL}(P|Q )$

(ref)[https://stats.stackexchange.com/questions/357963/what-is-the-difference-cross-entropy-and-kl-divergence]
[https://machinelearningmastery.com/cross-entropy-for-machine-learning/]
# Activation Function

## ReLU

解决梯度消失：相对于sigmoid和tanh，|输入值|较大时，导数很小

dying ReLU: 

假设一个神经元(neuron)是Relu(wx+b)Relu(wx+b)。因为一般用mini batch（SGD）优化算法，每次计算gradient只用一组（batch）数据点。假如用一组数据点更新w,bw,b后，其余数据点wx+b<0wx+b<0，那么只会有一组点能通过这个neuron并更新它的参数，对于绝大多数点来说，不能通过这个neuron，也不能更新参数，相当于“死掉”。如果dying relu 很多，对于大多数数据来说神经网络大部分通路断掉，学习能力就变弱了。

解决办法是用leakyRelu等；bias的初始值设为正数，比如1；减小learning rate。

## Leaky ReLU
# IoU compute
``` 
box1=[x1,y1,x2,y2]
box2=[x3,y3,x4,y4]
tlx=max(x1,x3)
tly=max(y1,y3)
brx=min(x2,x4)
bry=min(y2,y4)
w=max(brx-tlx,0)
h=max(bry-tly,0)

IoU=w*h/(S1+S2-w*h)
```
# L_0, L_1, L_2, L_p
## L_0
### L_0 不是norm
Definition of norm:
1. homogeneous: $ \forall x \in \mathcal{R}^n,k\in \mathbb{R},\lVert kx\|=|k|\|x\| $
2. triangle inequality
3. positive define, $ if \|x\|=0,x=0$
$L_0$不符合1，所以不是norm
### 为什么不用L_0做优化
首先L0缺少数学表示，使得L0的优化问题是NPhard，于是在相关领域（压缩感知）的优化问题中，会转化成L1，因为l1是the best convexification of the L0 "norm" ("the closest convex relaxation of best subset selection")
## L_1
优：解的稀疏性，从解空间解释
弊：计算较昂贵，并不是处处可导
## L_2
优：算的快
弊：不洗漱
# YoLo
## v1
1. anchor free
2. 图片切分为7 * 7
3. backbone 24层
4. 224训backbone, 448训练predictor
5. loss针对bbox的中心，高度和宽度、置信度、grid里是否有object
优点：
1. 相比proposal-based方法，速度很快
缺点：
1. 无法很好的检测较小的群聚目标
2. 对图片输入尺寸有限制
## v2
1. bn
2. 224训backbon后448微调backbone，输入边长两倍
3. anchor box，有kmeans得到
4. 针对小目标：由7 * 7升为13 * 13，预测框大约900
5. passthrough，最后的pooling前将featuremap拆分，并concatenate![passthrough](image/cv/1616990227702.png)
6. 多尺度训练，去掉了fc
7. word tree利用分类数据训练检测模型
8. 最终输出维度13* 13 * 5 * 25，其中13代表网格数，5代表anchor数量，25=（20类+4坐标+1置信度） ![](image/cv/1616992630816.png)
## v3
1. darknet 53，有残差
2. 多尺度预测。在3个大小的feature map进行预测![](image/cv/1616991004727.png)
3. 预测框很多，10000左右，所以比v2慢
## v4
1. 数据增强，除了经典的几何畸变与光照畸变外，还创新地使用了图像遮挡(**Random Erase，Cutout，Hide and Seek，Grid Mask ，MixUp**)技术，对于多图组合，作者混合使用了**CutMix与Mosaic** 技术。除此之外，作者还使用了**Self-Adversarial Training (SAT)**来进行数据增强。
2. backbone: cspdarknet
3. panet
4. mish激活函数，计算较慢

## v5
1. **缩放，色彩空间调整和马赛克增强**。
2. 自适应anchor box，基于数据集自动学习
3. cspdarknet
4. panet(path aggregation net)![](image/cv/1616992423423.png)