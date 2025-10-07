class MSELoss:
    """均方误差损失函数"""
    def __call__(self, predictions, targets):
        # loss = mean((predictions - targets)^2)
        return ((predictions - targets) * (predictions - targets)).sum()
    
class SGD:
    """自定义的随机梯度下降优化器"""
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """执行一次参数更新"""
        for param in self.parameters:
            # param.data = param.data - self.lr * param.grad
            param.data -= self.lr * param.grad

    def zero_grad(self):
        """清空所有参数的梯度"""
        for param in self.parameters:
            param.grad = 0.0