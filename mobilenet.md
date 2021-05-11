# MobileNet v1

## Depth-wise Convolution

常规卷积：对$H\times W \times C$进行就卷积，可以拓展通道数，参数量 $K_H \times K_W \times C_{in} \times C_{out}$

深度可分离卷积：每个卷积核对$H\times W$进行卷积，卷积核数量等于输入的通道数，无法拓展通道数，但是显著减少了参数量，参数量 $K_H \times K_W \times C_{in} $

## Point-wise Convolution

与常规卷积类似，卷积核尺寸为$ 1\times 1$，对$ 1\times 1 \times C$

## Building Block

3,3 depthwise conv -> BN -> ReLU -> 1,1 Conv -> BN -> ReLU

## [ref]

[link](https://zhuanlan.zhihu.com/p/92134485)

# MobileNet v2

## Inverted Residuals

引入残差，先升维再降维，与resnet沙漏形相反，所以称为inverted

## Linear Bottleneck

将narrow layer后的relu去除。作者认为非线性层会在高维结构能增加非线性，但在低维空间会破坏特征

## 全卷积网络


## Building Block

1,1 pw conv -> ReLU 6-> 3,3 dw conv -> ReLU 6-> 1,1 pw Conv -> BN
