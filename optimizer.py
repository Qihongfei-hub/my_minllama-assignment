from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-5,   ## qhf
            betas: Tuple[float, float] = (0.9, 0.999),
            # AdamW 优化器的一个重要超参数,
            #第一个值 (0.9) ：梯度一阶矩（first moment）的指数移动平均（EMA）衰减系数
            #第二个值 (0.999) ：梯度二阶矩（second moment）的指数移动平均衰减系数
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        # 参数验证
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        
        # 设置默认超参数
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """执行单个优化步骤"""
        loss = None
        if closure is not None:
            #大多数情况下，使用 AdamW 时不需要提供 closure
            loss = closure()

        # 遍历所有参数组
        for group in self.param_groups:
            # 遍历组内所有参数
            for p in group["params"]:
                # 跳过没有梯度的参数
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                # 检查梯度是否为稀疏张量
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # 状态字典，存储每个参数的历史信息
                state = self.state[p]

                # 初始化状态（首次更新时）
                if len(state) == 0:
                    # 初始化步骤计数器
                    state["step"] = 0
                    # 初始化梯度的指数移动平均值（一阶矩）
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # 初始化梯度平方的指数移动平均值（二阶矩）
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                # 从组字典中获取超参数
                alpha = group["lr"]  # 学习率
                beta1, beta2 = group["betas"]  # 一阶和二阶矩的衰减系数
                eps = group["eps"]  # 数值稳定性常数
                weight_decay = group["weight_decay"]  # 权重衰减系数
                correct_bias = group["correct_bias"]  # 是否修正偏差

                # 更新步骤计数器
                state["step"] += 1

                # 获取当前的一阶矩和二阶矩
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # 更新一阶矩和二阶矩
                # 一阶矩：exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 二阶矩：exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差修正
                # 参考论文：https://arxiv.org/abs/1412.6980
                if correct_bias:
                    # 一阶矩修正因子：1 - beta1^step
                    bias_correction1 = 1 - beta1 ** state["step"]
                    # 二阶矩修正因子：1 - beta2^step
                    bias_correction2 = 1 - beta2 ** state["step"]
                    # 计算修正后的学习率步长
                    step_size = alpha * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = alpha

                # 参数更新
                # p = p - step_size * exp_avg / (sqrt(exp_avg_sq) + eps)
                p.data.addcdiv_(exp_avg, torch.sqrt(exp_avg_sq) + eps, value=-step_size)

                # 应用权重衰减（AdamW 的关键改进）
                # 注意：权重衰减是独立于梯度更新的步骤
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-weight_decay * alpha)

        return loss