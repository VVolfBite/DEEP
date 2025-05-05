#[cfg(test)]
mod tests {
    use ff_ext::ff::Field;
    use goldilocks::GoldilocksExt2 as F;
    use crate::tensor::Tensor;
    use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
    use crate::Element;

    #[test]
    fn test_tensor_mle() {
        // 创建一个2x2的tensor，按列存储：[1,3]是第一列，[2,4]是第二列
        let tensor = Tensor::new(vec![2, 2], vec![Element::from(1), Element::from(3), Element::from(2), Element::from(4)]);
        
        // 将tensor转换为多线性扩展
        let mle = tensor.to_mle_flat::<F>();

        // 测试在布尔点评估
        let test_points = vec![
            vec![F::from(0u64), F::from(0u64)],  // 应该返回1
            vec![F::from(0u64), F::from(1u64)],  // 应该返回2
            vec![F::from(1u64), F::from(0u64)],  // 应该返回3
            vec![F::from(1u64), F::from(1u64)],  // 应该返回4
        ];

        println!("在布尔点上的评估结果：");
        for point in test_points.iter() {
            let result = mle.evaluate(point);
            println!("点 {:?}: {:?}", point, result);
        }

        // 测试固定第一个变量为1
        let fixed_point = vec![F::from(1u64)];  // 固定x₁=1
        let reduced_mle = mle.fix_variables(&fixed_point);

        // 在降维后的MLE上评估布尔点
        let reduced_test_points = vec![
            F::from(0u64),  // 应该返回3
            F::from(1u64),  // 应该返回4
        ];

        println!("\n固定第一个变量为1后的评估结果：");
        for point in reduced_test_points {
            let result = reduced_mle.evaluate(&[point]);
            println!("点 {:?}: {:?}", point, result);
        }
    }
} 