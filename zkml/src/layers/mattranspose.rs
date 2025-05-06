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

/// MatTranspose 结构体定义
pub struct MatTranspose<T> {
    matrix: Tensor<T>,
}

/// MatTransposeProof 结构体定义
pub struct MatTransposeProof<E: ExtensionField> {
    pub sumcheck: IOPProof<E>, // Sumcheck 证明
    pub evaluation: E,         // 评估值
}

impl<E: ExtensionField> MatTransposeProof<E> {
    /// 获取矩阵的评估值
    pub fn matrix_eval(&self) -> E {
        self.evaluation.clone()
    }
}

impl MatTranspose<Element> {
    /// 创建新的 MatTranspose 实例
    pub fn new(matrix: Tensor<Element>) -> Self {
        Self { matrix }
    }

    /// 执行矩阵转置操作
    pub fn op(&self) -> Tensor<Element> {
        self.matrix.transpose()
    }

    /// 提供辅助信息，用于虚拟多项式的构造
    pub fn aux_info<E: ExtensionField>(&self) -> VPAuxInfo<E> {
        let m_vars = self.matrix.nrows_2d().ilog2() as usize; // 行数变量
        let n_vars = self.matrix.ncols_2d().ilog2() as usize; // 列数变量
        VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![m_vars, n_vars]])
    }

    /// 在给定点评估矩阵
    pub fn evaluate_matrix_at<E: ExtensionField>(&self, point: &[E]) -> E {
        self.matrix.to_mle_2d().evaluate(point)
    }

    /// 证明函数
    pub fn prove<E, T>(
        &self,
        transcript: &mut T,
        last_claim: &Claim<E>,
    ) -> anyhow::Result<(MatTransposeProof<E>, Claim<E>)>
    where
        E: ExtensionField + Serialize + DeserializeOwned + Number,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
    {
        println!("\n--- MatTranspose Prove 函数开始执行 ---");

        // 获取矩阵维度
        let (m, n) = (self.matrix.nrows_2d(), self.matrix.ncols_2d());

        // 维度检查
        assert_eq!(
            m.ilog2() as usize + n.ilog2() as usize,
            last_claim.point.len(),
            "claim维度错误: 输出矩阵 {}x{} vs claim {}",
            m,
            n,
            last_claim.point.len()
        );

        println!("维度信息:");
        println!("  矩阵: {}x{}", m, n);
        println!("  last_claim point 长度: {}", last_claim.point.len());

        // 计算各维度的变量数量
        let m_vars = m.ilog2() as usize; // 行数变量
        let n_vars = n.ilog2() as usize; // 列数变量

        // 分割评估点，分别用于固定行和列
        // 注意：对于转置操作，我们需要交换行列变量的位置
        let row_points = &last_claim.point[..n_vars]; // 用于固定 j（原列）
        let col_points = &last_claim.point[n_vars..]; // 用于固定 i（原行）

        // 构建多线性扩展
        println!("\n--- 构建多线性扩展 ---");
        let mut matrix_mle = self.matrix.clone().to_mle_2d();
        println!("矩阵 MLE 信息:");
        println!("  - 原始变量数: {}", matrix_mle.num_vars());

        // 构建虚拟多项式
        println!("\n--- 构建虚拟多项式 ---");
        let mut vp = VirtualPolynomial::<E>::new(matrix_mle.num_vars());
        vp.add_mle_list(vec![matrix_mle.into()], E::ONE);

        // 执行证明
        println!("\n--- 执行证明过程 ---");
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, transcript);

        // 验证证明
        let output = self.op(); // 使用 op 方法执行矩阵转置
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
            eval: state.get_mle_final_evaluations()[0],
        };
        println!("Prove claim: point={:?}, eval={:?}", claim.point, claim.eval);

        let proof = MatTransposeProof {
            sumcheck: proof,
            evaluation: state.get_mle_final_evaluations()[0],
        };

        println!("--- MatTranspose Prove 函数执行完成 ---\n");
        Ok((proof, claim))
    }

    /// 验证函数
    pub fn verify<E: ExtensionField, T: Transcript<E>>(
        &self,
        transcript: &mut T,
        last_claim: &Claim<E>,
        proof: &MatTransposeProof<E>,
        aux_info: &VPAuxInfo<E>,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
    {
        println!("\n--- MatTranspose Verify 函数开始执行 ---");

        // 验证 Sumcheck 证明
        let subclaim =
            IOPVerifierState::<E>::verify(last_claim.eval, &proof.sumcheck, aux_info, transcript);
        println!("Sumcheck 验证通过");

        let matrix_point = subclaim.point_flat();
        println!("Matrix point: {:?}", matrix_point);

        // 验证矩阵的评估值
        let expected_eval = self.evaluate_matrix_at(&matrix_point);
        println!("矩阵评估值: expected={:?}, got={:?}", expected_eval, proof.matrix_eval());
        ensure!(
            expected_eval == proof.matrix_eval(),
            "矩阵评估值不一致: expected {:?}, got {:?}",
            expected_eval,
            proof.matrix_eval()
        );

        // 构造下一个 claim
        let next_claim = Claim {
            point: matrix_point,
            eval: proof.matrix_eval(),
        };

        println!("✅ MatTranspose Verify 函数执行完成");
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
    fn test_mattranspose_prove_verify() {
        println!("\n=== 开始矩阵转置测试 ===");

        // 1. 使用固定的矩阵数据
        let matrix_data = vec![
            Element::from(1),
            Element::from(2),
            Element::from(3),
            Element::from(4),
        ];

        let matrix = Tensor::<Element>::new(vec![2, 2], matrix_data).pad_next_power_of_two();
        let mattranspose = MatTranspose::new(matrix.clone());
        let output = mattranspose.op();

        println!("\n输入数据:");
        println!("原始矩阵: {:?}", matrix.get_data());
        println!("转置结果: {:?}", output.get_data());

        // 2. 使用固定的评估点
        let output_point = vec![F::from(0), F::from(1)];
        println!("评估点: {:?}", output_point);

        let output_claim = Claim::new(
            output_point.clone(),
            output.to_mle_flat().evaluate(&output_point),
        );
        println!("声明的评估值: {:?}", output_claim.eval);

        // 3. 执行证明
        let mut transcript = default_transcript::<F>();
        let (proof, claim) = mattranspose
            .prove(&mut transcript, &output_claim)
            .expect("证明生成失败");

        println!("\n验证过程:");
        let input_claim = mattranspose
            .verify(
                &mut default_transcript(),
                &output_claim,
                &proof,
                &mattranspose.aux_info(),
            )
            .expect("验证失败");
        println!("验证通过，新 claim: {:?}", input_claim);

        // 验证 Claim 是否一致
        assert_eq!(claim, input_claim, "Claim 不一致");

        println!("=== 矩阵转置测试完成 ===\n");
    }
} 