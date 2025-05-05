# 矩阵乘法证明实现指南

本文档详细说明了如何实现矩阵乘法证明，每个步骤都基于现有代码库中的实现。

## 1. 基本结构

### 1.1 上下文结构 (MatrixMulCtx)
```rust
pub struct MatrixMulCtx<F: ExtensionField> {
    sumcheck_aux: VPAuxInfo<F>,
    matrix_a_dims: Vec<usize>,
    matrix_b_dims: Vec<usize>,
}
```

参考来源：
- `zkml/src/layers/base.rs` 中的 `HadamardCtx` 结构
  - 使用 `sumcheck_aux` 进行验证
  - 使用 `VPAuxInfo` 存储验证信息
- `zkml/src/layers/matvec.rs` 中的 `MatVec` 结构
  - 矩阵维度信息的存储方式
  - 维度检查的实现

### 1.2 证明结构 (MatrixMulProof)
```rust
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct MatrixMulProof<F: ExtensionField> {
    sumcheck: IOPProof<F>,
    matrix_a_eval: F,
    matrix_b_eval: F,
    product_eval: F,
}
```

参考来源：
- `zkml/src/layers/base.rs` 中的 `HadamardProof` 结构
  - `sumcheck` 字段用于存储证明
  - 评估值的存储方式
- `zkml/src/layers/matvec.rs` 中的 `MatVecProof` 结构
  - 矩阵评估值的存储方式
  - 序列化/反序列化实现

## 2. 证明过程

### 2.1 证明函数 (prove)
```rust
pub fn prove<F: ExtensionField, T: Transcript<F>>(
    transcript: &mut T,
    matrix_a: &Tensor<Element>,
    matrix_b: &Tensor<Element>,
) -> MatrixMulProof<F>
```

实现步骤和参考来源：

1. 矩阵转换为 MLE 形式
   - 参考 `zkml/src/layers/base.rs` 中的 `to_mle_flat` 方法
   - 参考 `zkml/src/layers/matvec.rs` 中的 `to_mle_2d` 方法
   - 具体实现：
     ```rust
     let a_mle = matrix_a.to_mle_2d::<F>();
     let b_mle = matrix_b.to_mle_2d::<F>();
     ```

2. 生成随机评估点
   - 参考 `zkml/src/layers/base.rs` 中的 `transcript.read_challenge()` 方法
   - 具体实现：
     ```rust
     let eval_points = transcript.read_challenge().elements;
     ```

3. 计算评估值
   - 参考 `zkml/src/layers/base.rs` 中的 `evaluate` 方法
   - 参考 `zkml/src/layers/matvec.rs` 中的评估计算
   - 具体实现：
     ```rust
     let a_eval = a_mle.evaluate(&eval_points);
     let b_eval = b_mle.evaluate(&eval_points);
     ```

4. 构建虚拟多项式
   - 参考 `zkml/src/layers/base.rs` 中的 `VirtualPolynomial` 使用方式
   - 具体实现：
     ```rust
     let mut vp = VirtualPolynomial::<F>::new(a_mle.num_vars());
     vp.add_mle_list(vec![a_mle.into(), b_mle.into()], F::ONE);
     ```

## 3. 验证过程

### 3.1 验证函数 (verify)
```rust
pub fn verify<F: ExtensionField, T: Transcript<F>>(
    ctx: &MatrixMulCtx<F>,
    transcript: &mut T,
    proof: &MatrixMulProof<F>,
    expected_output: &Tensor<Element>,
) -> Result<bool>
```

实现步骤和参考来源：

1. 验证评估点
   - 参考 `zkml/src/layers/base.rs` 中的 `verify` 函数
   - 使用 `IOPVerifierState::verify` 进行验证
   - 具体实现：
     ```rust
     let subclaim = IOPVerifierState::<F>::verify(
         proof.product_eval,
         &proof.sumcheck,
         &ctx.sumcheck_aux,
         transcript,
     );
     ```

2. 验证乘积关系
   - 参考 `zkml/src/layers/base.rs` 中的乘积验证逻辑
   - 具体实现：
     ```rust
     let product = proof.matrix_a_eval * proof.matrix_b_eval;
     ensure!(product == proof.product_eval, "Product verification failed");
     ```

3. 验证输出矩阵
   - 参考 `zkml/src/layers/matvec.rs` 中的输出验证逻辑
   - 具体实现：
     ```rust
     let output_mle = expected_output.to_mle_2d::<F>();
     let output_eval = output_mle.evaluate(&proof.sumcheck.point);
     ensure!(output_eval == proof.product_eval, "Output verification failed");
     ```

## 4. 关键实现细节

### 4.1 MLE 转换
参考来源：
- `zkml/src/layers/matvec.rs` 中的 `to_mle_2d` 方法
- `zkml/src/layers/base.rs` 中的 `to_mle_flat` 方法

### 4.2 评估点生成
参考来源：
- `zkml/src/layers/base.rs` 中的 `read_challenge` 使用方式
- `zkml/src/layers/matvec.rs` 中的评估点生成

### 4.3 验证逻辑
参考来源：
- `zkml/src/layers/base.rs` 中的验证逻辑
- `zkml/src/layers/matvec.rs` 中的验证实现

## 5. 测试用例

### 5.1 基本测试
参考来源：
- `zkml/src/layers/base.rs` 中的测试用例结构
- `zkml/src/layers/matvec.rs` 中的测试实现

具体实现：
```rust
#[test]
fn test_matrix_mul_proving() {
    let mut transcript = default_transcript();
    let matrix_a = Tensor::random(&vec![n, k]);
    let matrix_b = Tensor::random(&vec![k, m]);
    // ... 测试逻辑
}
```

## 6. 注意事项

1. 维度检查
   - 参考 `zkml/src/layers/matvec.rs` 中的维度检查
   - 确保矩阵维度匹配

2. 评估点数量
   - 参考 `zkml/src/layers/base.rs` 中的评估点处理
   - 确保评估点数量与矩阵维度匹配

3. 验证顺序
   - 参考 `zkml/src/layers/base.rs` 和 `zkml/src/layers/matvec.rs` 的验证顺序
   - 先验证评估点
   - 再验证乘积关系
   - 最后验证输出矩阵

## 7. 实现建议

1. 分步实现
   - 先实现基本结构
   - 然后实现证明过程
   - 最后实现验证过程

2. 测试驱动
   - 为每个步骤编写测试用例
   - 确保与现有代码的兼容性

3. 性能优化
   - 参考 `zkml/src/layers/matvec.rs` 中的性能优化
   - 使用并行计算提高效率 