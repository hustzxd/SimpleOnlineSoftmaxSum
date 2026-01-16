import torch
import math

def offline_softmax(x, y):
    """
    Offline softmax: 一次性计算所有元素

    Args:
        x: 输入向量
        y: 权重向量

    Returns:
        score_sum: softmax 值的和（应该为1）
        xy_sum: softmax * y 的加权和
    """
    score = torch.softmax(x, dim=0)
    score_sum = score.sum().item()
    xy_sum = (score * y).sum().item()
    return score_sum, xy_sum


def online_softmax(x, y):
    """
    Online softmax: 流式处理，逐个元素计算

    Args:
        x: 输入向量
        y: 权重向量

    Returns:
        score_sum: softmax 值的和（应该为1）
        xy_sum: softmax * y 的加权和
    """
    m = -float("inf")   # running max
    l = 0.0             # log(Z) where Z = sum(exp(x_i - m))
    score_sum = 0.0     # softmax 累加和
    xy_sum = 0.0        # softmax * y 的累加和

    for i, xi in enumerate(x):
        m_new = max(m, xi.item())
        l_new = math.log(math.exp(l) * math.exp(m - m_new) + math.exp(xi - m_new))
        score_new = math.exp(xi - m_new) / math.exp(l_new)

        # 重标定公式
        scale = math.exp(m - m_new) * math.exp(l - l_new)
        score_sum = score_sum * scale + score_new
        xy_sum = xy_sum * scale + score_new * y[i].item()

        m = m_new
        l = l_new

    return score_sum, xy_sum


if __name__ == "__main__":
    # 固定随机种子
    torch.manual_seed(0)

    # 测试不同的shape
    shapes = [16, 32, 64, 128]
    tolerance = 1e-6
    all_passed = True

    print("=" * 70)
    print("测试不同 shape 的 Offline vs Online Softmax")
    print("=" * 70)

    for shape in shapes:
        print(f"\n{'='*70}")
        print(f"测试 shape = {shape}")
        print(f"{'='*70}")

        # 生成随机输入
        x = torch.randn(shape)
        y = torch.randn(shape)

        # 调用两个函数
        offline_score_sum, offline_xy_sum = offline_softmax(x, y)
        online_score_sum, online_xy_sum = online_softmax(x, y)

        # 计算差异
        score_diff = abs(offline_score_sum - online_score_sum)
        xy_diff = abs(offline_xy_sum - online_xy_sum)

        # 验证是否通过
        score_match = score_diff < tolerance
        xy_match = xy_diff < tolerance
        passed = score_match and xy_match

        # 输出结果
        print(f"\n[Offline] score_sum: {offline_score_sum:.10f}, xy_sum: {offline_xy_sum:.10f}")
        print(f"[Online]  score_sum: {online_score_sum:.10f}, xy_sum: {online_xy_sum:.10f}")
        print(f"\n差异:")
        print(f"  score_sum: {score_diff:.2e} {'✓' if score_match else '✗'}")
        print(f"  xy_sum:    {xy_diff:.2e} {'✓' if xy_match else '✗'}")
        print(f"\n结果: {'✓ 通过' if passed else '✗ 失败'}")

        if not passed:
            all_passed = False

    # 最终总结
    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")
    print(f"测试 shapes: {shapes}")
    if all_passed:
        print("\n✓ 所有测试通过！Offline 和 Online 结果一致。")
    else:
        print("\n✗ 部分测试失败！")
