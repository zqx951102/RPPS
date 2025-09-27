from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd

#定义了一个Cluster Proxy Memory模块，主要用于无监督/半监督学习或跨域场景中的特征聚类学习。
# 结合了**记忆机制（memory bank）与动量更新策略（momentum update）**来稳定地学习聚类中心。
class CM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features              # 特征矩阵，形状 [num_classes, feature_dim]
        ctx.momentum = momentum             # 动量因子（用于更新记忆中的类中心）
        ctx.save_for_backward(inputs, targets)    # 保存用于反向传播的张量
        outputs = inputs.mm(ctx.features.t()) # 计算输入 features 与类中心的相似度（矩阵乘法）

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):  #如果 inputs 需要梯度，就计算对 inputs 的梯度（即从 loss 反传回来）。
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
#记忆更新机制：对每个样本的目标类别y，用当前输入x更新类中心。 •使用 momentum 滑动平均方式更新（Exponential MovingAverage）。•再对新的类中心做归一化。
        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None  #只对 inputs 反传梯度，其它三个输入参数不需要梯度。


def cm(inputs, indexes, features, momentum=0.5):  #一个更易调用的封装函数，用于触发自定义前向/反向的 CM 类
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterProxyMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, source_classes, temp=0.05, momentum=0.2):
        super().__init__()
        #注册了一个memory bank：[num_samples, num_features]，代表每类的聚类中心/代理特征。
        self.num_features = num_features   # 每个特征向量维度
        self.num_samples = num_samples   # proxy memory 的总类数（或聚类数）
        self.source_classes = source_classes   # 来源域的类数（常用于 ignore_index）
        self.momentum = momentum        # proxy 更新的动量因子
        self.temp = temp                # softmax温度，用于控制 logits 的平滑度

        self.register_buffer("features", torch.zeros(num_samples, num_features))  #使用 register_buffer 表示这不是模型参数，但会随着模型保存/加载而保留

    def forward(self, inputs, targets, is_source=True):  #计算相似度并更新 memory
        targets = torch.cat(targets)     # 将 batch 内多个目标列表拼接成一个 long tensor
        targets = targets - 1    # # 将标签编号从 1-based 改为 0-based
        inds = targets >= 0    # 忽略非法标签（例如 -1）
        targets = targets[inds]
        for i in range(len(targets)):    # 特殊处理标签为 5554 的样本（可能是 placeholder），强制将它设为 source_classes - 1 类。这可能是为了使该类作为“ignore_index”的标签存在。
            if (targets[i] == 5554) and is_source:
                targets[i] = self.source_classes - 1

        valid_inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features) #过滤出有效的输入特征，即对应合法标签的样本
        outputs = cm(valid_inputs, targets, self.features, self.momentum)
#与 memory 中的特征计算相似度（dot product），再除以温度 temp 进行平滑。
        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets, ignore_index=self.source_classes - 1)  #使用交叉熵计算损失。特殊标签（如 source_classes - 1）不参与 loss 计算。

        return loss

# CM自定义 autograd 函数，实现动量更新聚类中心
# ClusterProxyMemory存储并更新 proxy 特征的模块，适用于无监督聚类或对抗学习中的目标增强
# forward()1. 过滤有效输入 2. 通过 cm() 与 memory 匹配 3. 计算 softmax+loss
