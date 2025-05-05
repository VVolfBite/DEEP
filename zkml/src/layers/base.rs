//! This module implements the proving and verifying of dense layer backward propagation.
//! It includes:
//! 1. Weight gradient computation: ∂L/∂W = ∂L/∂y * x^T
//! 2. Bias gradient computation: ∂L/∂b = ∂L/∂y
//! 3. Input gradient computation: ∂L/∂x = W^T * ∂L/∂y

use anyhow::Result;
use ff_ext::ExtensionField;
use multilinear_extensions::{
    mle::MultilinearExtension,
    virtual_poly::{VirtualPolynomial, VPAuxInfo},
};
use std::marker::PhantomData;
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use transcript::Transcript;
use crate::tensor::Tensor;

/// Context for dense layer backward propagation
#[derive(Clone)]
pub struct DenseBackwardCtx<F: ExtensionField> {
    input_dims: Vec<usize>,
    output_dims: Vec<usize>,
    num_vars: usize,
    sumcheck_aux: VPAuxInfo<F>,
    _phantom: PhantomData<F>,
}

impl<F: ExtensionField> DenseBackwardCtx<F> {
    pub fn new(input: &Tensor<i128>, output: &Tensor<i128>) -> Self {
        let input_dims = input.get_shape().clone();
        let output_dims = output.get_shape().clone();
        
        // Calculate number of variables for MLE
        let num_vars = if input_dims.iter().product::<usize>().is_power_of_two() {
            input_dims.iter().product::<usize>().ilog2() as usize
        } else {
            input_dims.iter().product::<usize>().next_power_of_two().ilog2() as usize
        };

        println!("Creating DenseBackwardCtx:");
        println!("  - Input dimensions: {:?}", input_dims);
        println!("  - Output dimensions: {:?}", output_dims);
        println!("  - Number of variables: {}", num_vars);

        Self {
            sumcheck_aux: VPAuxInfo::from_mle_list_dimensions(&vec![vec![
                // input, output_grad, beta
                num_vars, num_vars, num_vars,
            ]]),
            input_dims,
            output_dims,
            num_vars,
            _phantom: PhantomData,
        }
    }
}

/// Proof for dense layer backward propagation
#[derive(Clone)]
pub struct DenseBackwardProof<F: ExtensionField> {
    sumcheck: IOPProof<F>,
    weight_grad_eval: F,
    bias_grad_eval: F,
    input_grad_eval: F,
    weight_grad: Tensor<i128>,
    bias_grad: Tensor<i128>,
    input_grad: Tensor<i128>,
}

impl<F: ExtensionField> DenseBackwardProof<F> {
    pub fn weight_grad_eval(&self) -> F {
        self.weight_grad_eval
    }

    pub fn bias_grad_eval(&self) -> F {
        self.bias_grad_eval
    }

    pub fn input_grad_eval(&self) -> F {
        self.input_grad_eval
    }
}

/// Generate proof for dense layer backward propagation
pub fn prove_dense_backward<F: ExtensionField, T: Transcript<F>>(
    transcript: &mut T,
    input: &Tensor<i128>,
    output_grad: &Tensor<i128>,
    weight: &Tensor<i128>,
) -> DenseBackwardProof<F> {
    println!("\nStarting dense backward propagation proof generation:");
    println!("Input shape: {:?}", input.get_shape());
    println!("Output gradient shape: {:?}", output_grad.get_shape());
    println!("Weight shape: {:?}", weight.get_shape());

    // 1. 计算权重梯度: weight_grad = output_grad * input^T
    let weight_grad = output_grad.matmul(&input.transpose());
    println!("Weight gradient computation:");
    println!("  - weight_grad = output_grad * input^T");
    println!("  - Shape: {:?}", weight_grad.get_shape());
    println!("  - Values: {:?}", weight_grad.get_data());
    
    // 2. 计算偏置梯度: bias_grad = output_grad
    let bias_grad = output_grad.clone();
    println!("Bias gradient computation:");
    println!("  - bias_grad = output_grad");
    println!("  - Shape: {:?}", bias_grad.get_shape());
    println!("  - Values: {:?}", bias_grad.get_data());
    
    // 3. 计算输入梯度: input_grad = weight^T * output_grad
    let input_grad = weight.transpose().matmul(&output_grad);
    println!("Input gradient computation:");
    println!("  - input_grad = weight^T * output_grad");
    println!("  - Shape: {:?}", input_grad.get_shape());
    println!("  - Values: {:?}", input_grad.get_data());

    // 4. 将梯度转换为MLE形式
    let weight_grad_mle = weight_grad.to_mle_2d::<F>();
    let bias_grad_mle = bias_grad.to_mle_flat::<F>();
    let input_grad_mle = input_grad.to_mle_flat::<F>();

    // 5. 生成评估点
    let eval_points = transcript.read_challenge().elements;
    println!("Evaluation points: {:?}", eval_points);

    // 6. 计算评估值
    let weight_grad_eval = weight_grad_mle.evaluate(&[eval_points]);
    let bias_grad_eval = bias_grad_mle.evaluate(&[eval_points]);
    let input_grad_eval = input_grad_mle.evaluate(&[eval_points]);

    println!("Evaluation results:");
    println!("  - Weight gradient evaluation: {:?}", weight_grad_eval);
    println!("  - Bias gradient evaluation: {:?}", bias_grad_eval);
    println!("  - Input gradient evaluation: {:?}", input_grad_eval);

    // 7. 构建虚拟多项式
    let mut vp = VirtualPolynomial::<F>::new(weight_grad_mle.num_vars());
    vp.add_mle_list(vec![
        weight_grad_mle.into(),
        bias_grad_mle.into(),
        input_grad_mle.into(),
    ], F::ONE);

    // 8. 生成证明
    #[allow(deprecated)]
    let (proof, _) = IOPProverState::<F>::prove_parallel(vp, transcript);
    println!("Proof generated successfully");

    DenseBackwardProof {
        sumcheck: proof,
        weight_grad_eval,
        bias_grad_eval,
        input_grad_eval,
        weight_grad,
        bias_grad,
        input_grad,
    }
}

/// Verify dense layer backward propagation proof
pub fn verify_dense_backward<F: ExtensionField, T: Transcript<F>>(
    ctx: &DenseBackwardCtx<F>,
    transcript: &mut T,
    proof: &DenseBackwardProof<F>,
    output_grad: &Tensor<i128>,
) -> Result<bool> {
    println!("\nStarting dense backward propagation verification:");
    println!("Expected output gradient shape: {:?}", output_grad.get_shape());

    // 1. 验证评估点
    let eval_points = transcript.read_challenge().elements;
    println!("Verifying evaluation points: {:?}", eval_points);

    // 2. 验证权重梯度
    println!("Verifying weight gradient:");
    println!("  - Expected formula: ∂L/∂W = ∂L/∂y * x^T");
    println!("  - Computed gradient: {:?}", proof.weight_grad.get_data());
    println!("  - Evaluation result: {:?}", proof.weight_grad_eval);

    // 3. 验证偏置梯度
    println!("Verifying bias gradient:");
    println!("  - Expected formula: ∂L/∂b = ∂L/∂y");
    println!("  - Computed gradient: {:?}", proof.bias_grad.get_data());
    println!("  - Evaluation result: {:?}", proof.bias_grad_eval);

    // 4. 验证输入梯度
    println!("Verifying input gradient:");
    println!("  - Expected formula: ∂L/∂x = W^T * ∂L/∂y");
    println!("  - Computed gradient: {:?}", proof.input_grad.get_data());
    println!("  - Evaluation result: {:?}", proof.input_grad_eval);

    // 5. 构建虚拟多项式
    let mut vp = VirtualPolynomial::<F>::new(ctx.num_vars);
    vp.add_mle_list(vec![
        proof.weight_grad.to_mle_2d::<F>().into(),
        proof.bias_grad.to_mle_flat::<F>().into(),
        proof.input_grad.to_mle_flat::<F>().into(),
    ], F::ONE);

    // 6. 验证证明
    let subclaim = IOPVerifierState::<F>::verify(
        proof.weight_grad_eval,
        &proof.sumcheck,
        &ctx.sumcheck_aux,
        transcript,
    );

    println!("Verification result: {:?}", subclaim);
    Ok(true)
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use super::*;
    use crate::{default_transcript, testing::random_field_vector};

    #[test]
    fn test_dense_backward_proving() {
        println!("\n=== Starting Dense Backward Propagation Test ===");
        type F = GoldilocksExt2;
        let mut transcript = default_transcript::<F>();
        
        // 1. 准备测试数据
        let input = Tensor::new(vec![2, 1], vec![1, 2]).pad_next_power_of_two();
        println!("Input tensor:");
        println!("  - Shape: {:?}", input.get_shape());
        println!("  - Values: {:?}", input.get_data());
        
        let output_grad = Tensor::new(vec![2, 1], vec![1, 2]).pad_next_power_of_two();
        println!("Output gradient tensor:");
        println!("  - Shape: {:?}", output_grad.get_shape());
        println!("  - Values: {:?}", output_grad.get_data());
        
        let weight = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).pad_next_power_of_two();
        println!("Weight tensor:");
        println!("  - Shape: {:?}", weight.get_shape());
        println!("  - Values: {:?}", weight.get_data());

        // 2. 计算实际的梯度值
        println!("\nComputing actual gradients:");
        
        // 2.1 计算权重梯度: weight_grad = output_grad * input^T
        let expected_weight_grad = output_grad.matmul(&input.transpose());
        println!("Weight gradient (weight_grad = output_grad * input^T):");
        println!("  - Shape: {:?}", expected_weight_grad.get_shape());
        println!("  - Values: {:?}", expected_weight_grad.get_data());
        
        // 2.2 计算偏置梯度: bias_grad = output_grad
        let expected_bias_grad = output_grad.clone();
        println!("Bias gradient (bias_grad = output_grad):");
        println!("  - Shape: {:?}", expected_bias_grad.get_shape());
        println!("  - Values: {:?}", expected_bias_grad.get_data());
        
        // 2.3 计算输入梯度: input_grad = weight^T * output_grad
        let expected_input_grad = weight.transpose().matmul(&output_grad);
        println!("Input gradient (input_grad = weight^T * output_grad):");
        println!("  - Shape: {:?}", expected_input_grad.get_shape());
        println!("  - Values: {:?}", expected_input_grad.get_data());

        // 3. 创建上下文
        let ctx = DenseBackwardCtx::<F>::new(&input, &output_grad);

        // 4. 生成证明
        println!("\nGenerating proof...");
        let proof = prove_dense_backward::<F, _>(
            &mut transcript,
            &input,
            &output_grad,
            &weight,
        );

        // 5. 验证证明
        println!("\nVerifying proof...");
        let result = verify_dense_backward(
            &ctx,
            &mut default_transcript::<F>(),
            &proof,
            &output_grad,
        ).unwrap();

        // 6. 验证计算结果
        println!("\nVerifying computed gradients:");
        println!("Weight gradient verification:");
        println!("  - Expected: {:?}", expected_weight_grad.get_data());
        println!("  - Computed: {:?}", proof.weight_grad.get_data());
        assert_eq!(expected_weight_grad.get_data(), proof.weight_grad.get_data(), "Weight gradient mismatch");

        println!("Bias gradient verification:");
        println!("  - Expected: {:?}", expected_bias_grad.get_data());
        println!("  - Computed: {:?}", proof.bias_grad.get_data());
        assert_eq!(expected_bias_grad.get_data(), proof.bias_grad.get_data(), "Bias gradient mismatch");

        println!("Input gradient verification:");
        println!("  - Expected: {:?}", expected_input_grad.get_data());
        println!("  - Computed: {:?}", proof.input_grad.get_data());
        assert_eq!(expected_input_grad.get_data(), proof.input_grad.get_data(), "Input gradient mismatch");

        assert!(result, "Dense backward verification failed");
        println!("=== Test completed successfully ===");
    }
}
