use anyhow::ensure;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use transcript::Transcript;

use crate::{
    Claim, Element, Tensor,
    quantization::{Fieldizer, TensorFielder},
    tensor::Number,
};

/// MatMat 结构体定义
pub struct MatMat<T> {
    matrix_left: Tensor<T>,
}

/// MatMatProof 结构体定义
pub struct MatMatProof<E: ExtensionField> {
    pub sumcheck: IOPProof<E>, // Sumcheck 证明
    pub evaluations: Vec<E>,   // 评估值
}

impl<E: ExtensionField> MatMatProof<E> {
    /// 获取左矩阵的评估值
    pub fn left_matrix_eval(&self) -> E {
        self.evaluations[0].clone()
    }

    /// 获取右矩阵的评估值
    pub fn right_matrix_eval(&self) -> E {
        self.evaluations[1].clone()
    }
}

impl MatMat<Element> {
    /// 创建新的 MatMat 实例
    pub fn new(matrix_left: Tensor<Element>) -> Self {
        Self { matrix_left }
    }

    /// 执行矩阵乘法操作
    pub fn op(&self, matrix_right: &Tensor<Element>) -> Tensor<Element> {
        self.matrix_left.matmul(matrix_right)
    }

    /// 提供辅助信息，用于虚拟多项式的构造
    pub fn aux_info<E: ExtensionField>(&self) -> VPAuxInfo<E> {
        let k_vars = self.matrix_left.ncols_2d().ilog2() as usize; // 中间维度变量数
        VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![k_vars, k_vars]])
    }

    /// 在给定点评估矩阵
    pub fn evaluate_matrix_at<E: ExtensionField>(&self, point: &[E]) -> E {
        self.matrix_left.to_mle_2d().evaluate(point)
    }

    /// 证明函数
    pub fn prove<E, T>(
        &self,
        transcript: &mut T,
        last_claim: &Claim<E>,
        matrix_right: &Tensor<Element>,
    ) -> anyhow::Result<(MatMatProof<E>, Claim<E>)>
    where
        E: ExtensionField + Serialize + DeserializeOwned + Number,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
    {
        println!("\n--- MatMat Prove 函数开始执行 ---");

        // 打印矩阵的形状
        println!("左矩阵形状: {:?}", self.matrix_left.get_shape());
        println!("右矩阵形状: {:?}", matrix_right.get_shape());

        // 获取矩阵维度
        let (m, k1) = (self.matrix_left.nrows_2d(), self.matrix_left.ncols_2d());
        let (k2, n) = (matrix_right.nrows_2d(), matrix_right.ncols_2d());

        // 打印维度信息
        println!("左矩阵维度: m = {}, k1 = {}", m, k1);
        println!("右矩阵维度: k2 = {}, n = {}", k2, n);

        // 维度检查
        assert_eq!(k1, k2, "矩阵维度不匹配: {}x{} @ {}x{}", m, k1, k2, n);
        assert_eq!(
            m.ilog2() as usize + n.ilog2() as usize,
            last_claim.point.len(),
            "claim维度错误: 输出矩阵 {}x{} vs claim {}",
            m,
            n,
            last_claim.point.len()
        );

        println!("维度信息:");
        println!("  左矩阵: {}x{}", m, k1);
        println!("  右矩阵: {}x{}", k2, n);
        println!("  last_claim point 长度: {}", last_claim.point.len());

        // 计算各维度的变量数量
        let m_vars = m.ilog2() as usize; // 行数变量
        let n_vars = n.ilog2() as usize; // 列数变量
        let k_vars = k1.ilog2() as usize; // 中间维度变量

        // 打印变量数量
        println!("变量数量:");
        println!("  m_vars (行数变量): {}", m_vars);
        println!("  n_vars (列数变量): {}", n_vars);
        println!("  k_vars (中间维度变量): {}", k_vars);
        // 分割评估点，分别用于固定行和列
        println!("last_claim.point.len(): {}", last_claim.point.len());

        let col_points = &last_claim.point[..n_vars]; // 用于固定 i（左矩阵的行）
        let row_points = &last_claim.point[n_vars..]; // 用于固定 j（右矩阵的列）
        // 打印分割后的评估点
        println!("row_points (用于固定 j): {:?}", row_points);
        println!("col_points (用于固定 i): {:?}", col_points);
        println!("\n--- 构建多线性扩展 ---");

        // 对左矩阵 A[i,k]：固定 i，保留 k → 固定低位变量
        let mut left_mle = self.matrix_left.clone().to_mle_2d();
        println!("左矩阵 MLE 信息:");
        println!("  - 原始变量数: {}", left_mle.num_vars());
        left_mle.fix_high_variables_in_place(row_points); // ✅ 改成 fix_variables_in_place
        println!("  - 固定行后变量数: {}", left_mle.num_vars());

        // 对右矩阵 B[k,j]：固定 j，保留 k → 固定高位变量
        let mut right_mle = matrix_right.clone().to_mle_2d();
        println!("右矩阵 MLE 信息:");
        println!("  - 原始变量数: {}", right_mle.num_vars());
        right_mle.fix_variables_in_place(col_points); // ✅ 改成 fix_high_variables_in_place
        println!("  - 固定列后变量数: {}", right_mle.num_vars());

        // 检查两个 MLE 的变量数是否匹配
        assert_eq!(
            left_mle.num_vars(),
            right_mle.num_vars(),
            "MLE 变量数不匹配: left={}, right={}",
            left_mle.num_vars(),
            right_mle.num_vars()
        );

        // 打印每个布尔输入点对应的 MLE 计算值
        println!("\n--- 打印 MLE 评估值 ---");
        let all_binary_vectors = |num_vars: usize| -> Vec<Vec<E>> {
            (0..(1 << num_vars))
                .map(|i| {
                    (0..num_vars)
                        .map(|j| if (i & (1 << j)) != 0 { E::ONE } else { E::ZERO })
                        .collect()
                })
                .collect()
        };

        let left_points = all_binary_vectors(left_mle.num_vars());
        let right_points = all_binary_vectors(right_mle.num_vars());

        for point in &left_points {
            println!("左矩阵 MLE @ {:?} = {:?}", point, left_mle.evaluate(point));
        }

        for point in &right_points {
            println!("右矩阵 MLE @ {:?} = {:?}", point, right_mle.evaluate(point));
        }

        // 构建虚拟多项式
        println!("\n--- 构建虚拟多项式 ---");
        let mut vp = VirtualPolynomial::<E>::new(k_vars);
        vp.add_mle_list(vec![left_mle.into(), right_mle.into()], E::ONE);

        // 执行证明
        println!("\n--- 执行证明过程 ---");
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, transcript);

        // 验证证明
        let output = self.op(matrix_right); // 使用 op 方法执行矩阵乘法
        let output_mle = output.to_mle_flat();
        let all_points = all_binary_vectors(output_mle.num_vars());
        for point in &all_points {
            println!(
                "Output MLE @ {:?} = {:?}",
                point,
                output_mle.evaluate(point)
            );
        }

        let claimed_sum = output.to_mle_flat().evaluate(&last_claim.point);
        println!("Claim Point: {:?}", last_claim.point);
        println!("Claimed Sum (from output.to_mle_flat()): {:?}", claimed_sum);

        println!("\n--- 证明验证 ---");
        println!("计算得到的和: {:?}", proof.extract_sum());
        println!("声明的和: {:?}", claimed_sum);
        println!("是否相等: {}", proof.extract_sum() == claimed_sum);
        println!("Proof 生成完成: {:?}", proof);

        debug_assert!(proof.extract_sum() == claimed_sum, "无效的 sumcheck 证明");

        // 构建返回值
        let claim = Claim {
            point: proof.point.clone(),
            eval: state.get_mle_final_evaluations()[1],
        };

        let proof = MatMatProof {
            sumcheck: proof,
            evaluations: state.get_mle_final_evaluations(),
        };

        println!("--- MatMat Prove 函数执行完成 ---\n");
        Ok((proof, claim))
    }
    /// 验证函数
    pub fn verify<E: ExtensionField, T: Transcript<E>>(
        &self,
        transcript: &mut T,
        last_claim: &Claim<E>,
        proof: &MatMatProof<E>,
        aux_info: &VPAuxInfo<E>,
        // 提供右矩阵在指定点处的评估值函数
        eval_right_matrix_at: impl FnOnce(&[E]) -> E,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
    {
        println!("\n--- MatMat Verify 函数开始执行 ---");

        // 验证 Sumcheck 证明
        let subclaim =
            IOPVerifierState::<E>::verify(last_claim.eval, &proof.sumcheck, aux_info, transcript);
        println!("Sumcheck 验证通过");

        let matrix_point = subclaim.point_flat();
        ensure!(
            matrix_point.len() == self.matrix_left.ncols_2d().ilog2() as usize,
            "Matrix point 长度不正确"
        );

        // 验证左矩阵的评估值
        let expected_left = self.matrix_left.to_mle_2d().evaluate(&matrix_point);
        ensure!(
            expected_left == proof.left_matrix_eval(),
            "左矩阵评估值不一致: expected {:?}, got {:?}",
            expected_left,
            proof.left_matrix_eval()
        );

        // 使用回调函数评估右矩阵
        let expected_right = eval_right_matrix_at(&matrix_point);
        ensure!(
            expected_right == proof.right_matrix_eval(),
            "右矩阵评估值不一致: expected {:?}, got {:?}",
            expected_right,
            proof.right_matrix_eval()
        );

        // 构造下一个 claim（将左矩阵评估作为新的 eval）
        let next_claim = Claim {
            point: matrix_point,
            eval: proof.left_matrix_eval(),
        };

        println!("✅ MatMat Verify 函数执行完成");
        Ok(next_claim)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{default_transcript, testing::random_field_vector};
    use goldilocks::GoldilocksExt2;

    type F = GoldilocksExt2;

    #[test]
    fn test_matmat_prove_verify() {
        println!("\n=== 开始矩阵乘矩阵证明测试 ===");

        // 1. 原始正确的计算
        let matrix_left = Tensor::<Element>::random_seed(&vec![5, 2], None).pad_next_power_of_two();
        let matrix_right =
            Tensor::<Element>::random_seed(&vec![2, 3], None).pad_next_power_of_two();
        let matmat = MatMat::new(matrix_left.clone());
        let output = matmat.op(&matrix_right);

        println!("\n=== 输入数据 ===");
        println!("左矩阵: {:?}", matrix_left.get_data());
        println!("右矩阵: {:?}", matrix_right.get_data());
        println!("\n=== 输出结果 ===");
        println!("正确输出: {:?}", output.get_data());

        // 2. 创建评估点和声明
        let output_point = random_field_vector(output.get_data().len().ilog2() as usize);
        println!("评估点: {:?}", output_point);

        let output_claim = Claim::new(
            output_point.clone(),
            output.to_mle_flat().evaluate(&output_point),
        );
        println!("声明的评估值: {:?}", output_claim.eval);

        // 3. 执行证明
        let mut transcript = default_transcript::<F>();
        let (proof, claim) = matmat
            .prove(&mut transcript, &output_claim, &matrix_right)
            .expect("证明生成失败");

        println!("\n=== 验证过程 ===");

        // 验证右矩阵的评估值
        let (k2, n) = (matrix_right.nrows_2d(), matrix_right.ncols_2d());
        let n_vars = n.ilog2() as usize; // 列数变量
        let col_points = &output_claim.point[..n_vars]; // 用于固定 i（左矩阵的行）
        let mut right_mle = matrix_right.clone().to_mle_2d();
        right_mle.fix_variables_in_place(col_points); // ✅ 改成 fix_high_variables_in_place
        let matrix_right_eval = right_mle.evaluate(&claim.point);
        println!("右矩阵评估值: {:?}", matrix_right_eval);
        assert_eq!(matrix_right_eval, claim.eval, "右矩阵评估值不匹配");

        let input_claim = matmat
            .verify(
                &mut default_transcript(),
                &output_claim,
                &proof,
                &matmat.aux_info(),
                |point| matmat.evaluate_matrix_at(point), // 传入评估函数
            )
            .expect("验证失败");
        println!("验证通过，新 Claim: {:?}", input_claim);

        // 验证 Claim 是否一致
        assert_eq!(claim, input_claim, "Claim 不一致");

        println!("=== 矩阵乘矩阵证明测试完成 ===");
    }
}
