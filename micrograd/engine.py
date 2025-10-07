import numpy as np

class Tensor:
    """ A minimal NumPy-based autograd system (multi-dimensional version of Tensor) """

    def __init__(self : 'Tensor', data: np.ndarray, _children: tuple = (), _op: str = '', requires_grad: bool = True) -> None:
        # 保证 data 一定是 ndarray
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=float)
        elif isinstance(data, (list, tuple)):
            data = np.array(data, dtype=float)

        self.data = data
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad

        # 构建计算图
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __getitem__(self, key):
        """
        实现索引功能，使得 tensor[key] 可以工作。
        返回一个新的 Tensor，共享底层数据，但具有独立的梯度。
        """
        # 使用 key 索引底层的 data 数组
        new_data = self.data[key]
        
        # 创建一个新的 Tensor，它的数据是索引后的结果
        # 它是否需要求导，取决于原始 Tensor 是否需要求导
        new_tensor = Tensor(new_data, requires_grad=self.requires_grad)
        
        return new_tensor

    def __add__(self : 'Tensor', other : 'Tensor') -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward() -> 'Tensor':
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward() -> 'Tensor':
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self: 'Tensor', other: float) -> 'Tensor':
        assert isinstance(other, (int, float)), "only support int/float power"
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward() -> 'Tensor':
            if self.requires_grad:
                self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __matmul__(self: 'Tensor', other: 'Tensor'):  # self @ other
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward() -> 'Tensor':
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    
    def __format__(self, format_spec):
        """
        重写格式化方法，使得 f"{tensor:.4f}" 能够工作。
        它会将格式化操作代理给底层的 data 属性。
        """
        # 检查数据是否是标量 (0维数组)
        if self.data.ndim == 0:
            # 如果是标量，直接对其进行格式化
            return self.data.__format__(format_spec)
        else:
            # 如果是数组，可以选择返回其字符串表示，或者像NumPy那样处理
            # 这里我们简单地返回其字符串表示
            return self.data.__repr__()

    def sum(self):
        """
        计算张量所有元素的总和，并返回一个新的 Tensor。
        同时设置反向传播函数，以便在 backward() 时将梯度均匀分配给所有原始元素。
        """
        # 1. 计算数据的总和
        out_data = self.data.sum()

        # 2. 创建一个新的 Tensor 来存储结果
        # 新 Tensor 是否需要求导，取决于原始 Tensor
        out = Tensor(out_data, requires_grad=self.requires_grad)

        # 3. 如果需要求导，定义反向传播函数
        if out.requires_grad:
            # 保存对原始 Tensor 的引用
            # 注意：这里使用 nonlocal 或一个可变对象来在闭包中存储 self
            # 一个简单的方法是使用一个列表来“捕获”self
            original_self = self
            
            def backward_sum():
                # 对于 sum 操作，上游传来的梯度（out.grad）需要均匀地分配给所有输入元素
                # 例如，如果 sum(x) = y，那么 dy/dx_i = 1 for all i
                # 所以，梯度是一个和 original_self.data 形状相同的全 1 数组，乘以 out.grad
                if original_self.requires_grad:
                    grad = np.ones_like(original_self.data) * out.grad
                    if original_self.grad is None:
                        original_self.grad = grad
                    else:
                        original_self.grad += grad
            
            # 将这个反向传播函数存储在输出 Tensor 的 _ctx 中
            out._ctx = backward_sum
            
        return out

    def relu(self: 'Tensor') -> 'Tensor':
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward() -> 'Tensor':
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
    
    # TODO: 改架构吧
    def conv1d_forward(x_data, w_data, stride=1):
        """
        一个非常简化的1D卷积前向计算。
        x_data: (batch_size, in_channels, width)
        w_data: (out_channels, in_channels, kernel_size)
        """
        batch_size, in_channels, width = x_data.shape
        out_channels, _, kernel_size = w_data.shape
        
        # 计算输出宽度
        out_width = (width - kernel_size) // stride + 1
        
        # 初始化输出
        output = np.zeros((batch_size, out_channels, out_width))
        
        # 执行卷积
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_width):
                    start = i * stride
                    end = start + kernel_size
                    # 从输入中提取感受野
                    receptive_field = x_data[b, :, start:end]
                    # 与卷积核进行点积
                    output[b, oc, i] = np.sum(receptive_field * w_data[oc, :, :])
                    
        return output

    def conv1d(self, other, stride=1):
        """
        other: 卷积核，是一个 Tensor 对象。
        """
        # 1. 执行数值计算
        out_data = self.conv1d_forward(self.data, other.data, stride=stride)
        
        # 2. 创建新的 Tensor 对象
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)
        
        # 3. 记录父节点
        out._prev.add(self)
        out._prev.add(other)
        
        # 4. 定义反向传播函数
        def _backward():
            # 我们需要闭包来捕获当前的 self, other, stride 等值
            x, w = self, other
            
            # 检查是否需要计算梯度
            if not (x.requires_grad or w.requires_grad):
                return
            
            # 获取输出的梯度
            dout = out.grad
            
            # 计算输入 x 的梯度 (dx) 和权重 w 的梯度 (dw)
            # 这需要一个卷积的反向传播函数
            def conv1d_backward(x_data, w_data, dout_data, stride=1):
                """
                简化的1D卷积反向传播。
                """
                batch_size, in_channels, width = x_data.shape
                out_channels, _, kernel_size = w_data.shape
                _, _, out_width = dout_data.shape
                
                # 初始化梯度为0
                dx_data = np.zeros_like(x_data)
                dw_data = np.zeros_like(w_data)
                
                # 计算权重梯度 dw
                # dw 的计算是输入数据 x 和输出梯度 dout 的“全卷积”
                for oc in range(out_channels):
                    for ic in range(in_channels):
                        for k in range(kernel_size):
                            # 累加所有批次和所有位置的贡献
                            for b in range(batch_size):
                                for i in range(out_width):
                                    # 权重 w[oc, ic, k] 参与了计算 out[b, oc, i]
                                    # 贡献为 x[b, ic, i*stride + k] * dout[b, oc, i]
                                    dw_data[oc, ic, k] += x_data[b, ic, i * stride + k] * dout_data[b, oc, i]
                
                # 计算输入梯度 dx
                # dx 的计算是输出梯度 dout 和权重 w 的“反卷积”或“全卷积”
                for b in range(batch_size):
                    for ic in range(in_channels):
                        for i in range(width):
                            # 找到所有会影响 x[b, ic, i] 的输出位置
                            for oc in range(out_channels):
                                for k in range(kernel_size):
                                    # 如果 x 的第 i 个元素在卷积核的第 k 个位置被使用
                                    # 那么它对应的输出位置是 j = i - k
                                    j = i - k
                                    if j >= 0 and j < out_width and (j * stride + k) == i:
                                        dx_data[b, ic, i] += w_data[oc, ic, k] * dout_data[b, oc, j]
                                        
                return dx_data, dw_data
            
            dx_data, dw_data = conv1d_backward(x.data, w.data, dout, stride=stride)
            
            # 将计算出的梯度累加到父节点的 .grad 属性上
            if x.requires_grad:
                if x.grad is None:
                    x.grad = dx_data
                else:
                    x.grad += dx_data
                    
            if w.requires_grad:
                if w.grad is None:
                    w.grad = dw_data
                else:
                    w.grad += dw_data

        out._backward = _backward
        return out

    def backward(self: 'Tensor') -> None:
        topo = []
        visited = set()

        def build_topo(v: 'Tensor') -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    # 其他运算
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    def __repr__(self: 'Tensor') -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"
