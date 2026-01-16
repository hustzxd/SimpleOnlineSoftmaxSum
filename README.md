# Online Softmax SUM

本项目实现了 offline softmax 和 online softmax 两种算法，并验证了流式处理中 softmax 计算的正确性。

## 背景

Softmax 是深度学习中常用的归一化函数：

$$
\text{softmax}(x_i) = \frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
$$

在离线（offline）场景中，可以一次性访问所有数据来计算 softmax。但在流式（online）场景中，数据逐个到达，无法回头访问之前的数据，因此需要一种特殊方法来维护正确的 softmax 值。

## 核心算法

### Offline Softmax SUM

```python
score = torch.softmax(x, dim=0)
xy_sum = (score * y).sum()
```

### Online Softmax SUM

使用重标定技术，维护三个状态变量：

- `m`: running max（当前最大值）
- `l`: log-sum-exp，`l = log(sum(exp(x_i - m)))`
- `score_sum`: softmax 值的累加和（应为 1）

**重标定公式：**

LSE（log-sum-exp）的定义是：

$$
\begin{aligned}
\mathrm{LSE}
&= \ln \sum_i e^{x_i} \\
&= \ln \sum_i \left(e^{x_i - m} e^{m}\right) \\
&= \ln \left( e^{m} \sum_i e^{x_i - m} \right) \\
&= m + \ln \sum_i e^{x_i - m}
\end{aligned}
$$

这里 $m$ 是全局最大值，因此需要全局信息。我们定义 $l$，注意它已经不是标准的 LSE 了：

$$
l = \mathrm{LSE} - m = \ln \sum_i e^{x_i - m}
$$

根据 $l$ 可以得到：

$$
\text{softmax}(x_t | m_t, l_t) = \frac{e^{x_t-m_t}}{e^{l_t}}
$$

Online Softmax 的过程中，每次只能看到当前的最大值 $m_t$，也可以计算得到当前的 $l_t$，进而计算得到 $\text{softmax}(x_t|m_t,l_t)$。注意，这个 softmax 值是根据当前的 $m_t, l_t$ 计算得到的，未来 $m_t, l_t$ 会更新，因此需要根据更新后的 $m_{t+1}, l_{t+1}$ 对 $\text{softmax}(x_t|m_t,l_t)$ 进行重新缩放：

$$
\begin{aligned}
\text{softmax}(x_t|m_{t+1},l_{t+1})
&= \frac{e^{x_t-m_{t+1}}}{e^{l_{t+1}}} \\
&= \text{softmax}(x_t | m_t, l_t) e^{m_t - m_{t+1}} e^{l_t - l_{t+1}}
\end{aligned}
$$

因此，我们需要根据最新的 $x_{t+1}$，历史的 $m_t, l_t$，推导出 $m_{t+1}, l_{t+1}$：

$m$ 的更新很简单：

$$
m_{t+1} = \max(m_t, x_{t+1})
$$

$l$ 的更新：

$$
\begin{aligned}
l_{t+1}
&= \ln\left(\sum_i^{t+1} e^{x_i - m_{t+1}}\right) \\
&= \ln\left(\sum_i^{t} e^{x_i - m_t + m_t - m_{t+1}} + e^{x_{t+1}-m_{t+1}}\right) \\
&= \ln\left(\sum_i^t e^{x_i-m_t} e^{m_t-m_{t+1}}+e^{x_{t+1}-m_{t+1}}\right) \\
&= \ln\left(e^{l_t} \cdot e^{m_t - m_{t+1}} + e^{x_{t+1}-m_{t+1}}\right)
\end{aligned}
$$

因此，可以写成以下递推算法：

```python
for i, xi in enumerate(x):
    m_new = max(m, xi.item())
    l_new = log(exp(l) * exp(m - m_new) + exp(xi - m_new))
    score_new = exp(xi - m_new) / exp(l_new)

    scale = exp(m - m_new) * exp(l - l_new)
    score_sum = score_sum * scale + score_new
    xy_sum = xy_sum * scale + score_new * y[i]

    m = m_new
    l = l_new
```

**关键洞察：** 当新的最大值出现时，之前的 softmax 值需要按 `scale` 重标定。

## 运行

```bash
# 对比两种方法，验证多种 shape
python compare_softmax.py
```

## 验证结果

| Shape | score_sum 差异 | xy_sum 差异 | 结果 |
|-------|----------------|-------------|------|
| 16    | 2.22e-16       | 2.91e-08    | ✓    |
| 32    | 1.19e-07       | 8.51e-09    | ✓    |
| 64    | 6.66e-16       | 1.19e-08    | ✓    |
| 128   | 1.44e-15       | 5.15e-08    | ✓    |

## 应用场景

- **注意力机制**: Transformer 中的加权求和 `Σ softmax(q·k) * v`
- **流式数据处理**: 实时推理、增量计算
- **内存受限场景**: 无法存储全部历史数据
