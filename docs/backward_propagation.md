# 神经网络层的反向传播计算

本文档总结了各个神经网络层的反向传播计算过程。

## 1. Dense层

Dense层的反向传播包含以下核心计算：

1. 权重梯度计算：
   ```
   ∂L/∂W = ∂L/∂y * x^T
   ```
   - 需要矩阵乘法 (MatMat)
   - 需要矩阵转置 (MatTranspose)

2. 偏置梯度计算：
   ```
   ∂L/∂b = ∂L/∂y
   ```
   - 直接使用上游梯度

3. 输入梯度计算：
   ```
   ∂L/∂x = W^T * ∂L/∂y
   ```
   - 需要矩阵转置 (MatTranspose)
   - 需要矩阵乘法 (MatMat)

## 2. 激活层

激活层的反向传播计算：

1. ReLU激活函数：
   ```
   ∂L/∂x = ∂L/∂y * (x > 0)
   ```
   - 需要元素级乘法 (MatElemMul)
   - 需要比较操作 (Compare)

2. Sigmoid激活函数：
   ```
   ∂L/∂x = ∂L/∂y * sigmoid(x) * (1 - sigmoid(x))
   ```
   - 需要元素级乘法 (MatElemMul)
   - 需要sigmoid函数计算

## 3. 卷积层

卷积层的反向传播包含：

1. 权重梯度计算：
   ```
   ∂L/∂W = conv2d(∂L/∂y, x^T)
   ```
   - 需要卷积操作 (Conv2D)
   - 需要矩阵转置 (MatTranspose)

2. 偏置梯度计算：
   ```
   ∂L/∂b = sum(∂L/∂y)
   ```
   - 需要矩阵求和 (MatSum)

3. 输入梯度计算：
   ```
   ∂L/∂x = conv2d_transpose(∂L/∂y, W^T)
   ```
   - 需要转置卷积 (Conv2DTranspose)
   - 需要矩阵转置 (MatTranspose)

## 4. 池化层

池化层的反向传播：

1. MaxPooling：
   ```
   ∂L/∂x = upsample(∂L/∂y, mask)
   ```
   - 需要上采样操作 (Upsample)
   - 需要掩码操作 (Mask)

2. AveragePooling：
   ```
   ∂L/∂x = upsample(∂L/∂y / kernel_size^2)
   ```
   - 需要上采样操作 (Upsample)
   - 需要标量除法 (ScalarDiv)

## 需要实现的核心计算类型

基于以上分析，我们需要实现以下核心计算类型的证明：

1. 矩阵操作：
   - 矩阵乘法 (MatMat) - 已完成
   - 矩阵转置 (MatTranspose)
   - 矩阵加法 (MatAdd)
   - 矩阵元素乘法 (MatElemMul)
   - 矩阵求和 (MatSum)

2. 卷积操作：
   - 卷积 (Conv2D)
   - 转置卷积 (Conv2DTranspose)

3. 激活函数：
   - ReLU梯度
   - Sigmoid梯度

4. 池化操作：
   - MaxPooling梯度
   - AveragePooling梯度

5. 其他操作：
   - 比较操作 (Compare)
   - 上采样 (Upsample)
   - 掩码操作 (Mask)
   - 标量除法 (ScalarDiv) 