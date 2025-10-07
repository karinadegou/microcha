import random
from micrograd.engine import Tensor
import numpy as np

class Module:
    def __init__(self):
        # 1. 初始化两个字典，用于存储子模块和参数
        # _modules: 存储子模块，键是属性名（如 'conv1'），值是 Module 对象
        self._modules = {}
        # _parameters: 存储参数，键是属性名（如 'weight'），值是 Tensor 对象
        self._parameters = {}

    def __setattr__(self, name: str, value) -> None:
        """
        2. 重载属性赋值方法。
        这是实现自动管理的关键！当你执行 self.w = ... 时，这个方法会被调用。
        """
        # 检查值是否是一个 Module 的实例（即子模块）
        if isinstance(value, Module):
            self._modules[name] = value
            object.__setattr__(self, name, value) 
        # TODO: 赋值覆盖掉了__init__？
        
        # 检查值是否是一个 Tensor 的实例，并且需要计算梯度（即可训练参数）
        # 注意：这里我们简化了判断，PyTorch 用的是专门的 Parameter 类
        elif isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
            object.__setattr__(self, name, value) 
            
        # 如果是普通属性（如 int, str, 或不需要梯度的 Tensor），则按正常方式处理
        else:
            # 调用 object 的 __setattr__ 来避免无限递归
            object.__setattr__(self, name, value)

    def parameters(self):
        """
        3. 提供一个生成器，用于递归地获取所有可训练参数。
        """
        # a. 首先，生成自己的参数
        for param in self._parameters.values():
            yield param
            
        # b. 然后，递归地生成所有子模块的参数
        for module in self._modules.values():
            # 使用 yield from 来“扁平化”子模块生成器的输出
            yield from module.parameters()

    def __call__(self, *args, **kwargs):
        """
        4. 重载调用方法，让模块实例可以像函数一样被调用。
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        5. 定义前向传播接口。
        这个方法必须被子类重写，否则会抛出错误。
        """
        # inspect.stack()[0][3] 会返回当前函数名 'forward'
        raise NotImplementedError(f"Subclasses of Module must implement the 'forward' method.")

    def __repr__(self) -> str:
        """
        提供一个友好的字符串表示，方便调试。
        """
        # 获取类名
        cls_name = self.__class__.__name__
        
        # 构建属性字符串
        attrs = []
        # 添加参数
        for name, param in self._parameters.items():
            attrs.append(f"{name}=Tensor(shape={param.data.shape})")
        # 添加子模块
        for name, module in self._modules.items():
            attrs.append(f"{name}={module.__repr__()}")
            
        # 组合成 "Classname(attr1=..., attr2=...)" 的形式
        return f"{cls_name}({', '.join(attrs)})"


class Linear(Module):
    def __init__(self: 'Linear', nin: int, nout: int, nonlin: bool=True):
        """
        初始化一个标准的线性层。

        Args:
            nin: 输入特征的数量 (in_features)。
            nout: 输出特征的数量 (out_features)。
            nonlin: 是否使用非线性激活函数。
        """
        super().__init__()

        self.nin = nin
        self.nout = nout
        self.nonlin = nonlin

        # 优化：使用更优的参数初始化方法 (Kaiming/He Initialization)
        # 权重矩阵的形状是 (nin, nout)
        k = 1.0 / np.sqrt(nin)
        self.w = Tensor(np.random.uniform(-k, k, size=(nin, nout)), requires_grad=True)
        
        # 偏置向量的形状是 (nout,)
        self.b = Tensor(np.zeros((1, nout)), requires_grad=True)

    def forward(self: 'Linear', x: Tensor) -> Tensor:
        """
        定义前向传播。

        Args:
            x: 输入张量，形状为 (..., nin)。

        Returns:
            输出张量，形状为 (..., nout)。
        """
        # 执行线性变换：y = x @ w + b
        # 使用 @ 运算符，它会调用 Tensor 类的 __matmul__ 方法
        # 这才是标准的矩阵乘法
        out = x @ self.w + self.b

        # 如果需要，应用非线性激活函数
        if self.nonlin:
            out = out.relu()
            
        return out

    def __repr__(self):
        # 从 _parameters 字典中安全地获取 'w'
        # 使用 self._parameters.get('w')，如果 'w' 不存在，会返回 None，而不是报错
        weight_tensor = self._parameters.get('w')
        
        if weight_tensor is not None:
            in_features = len(weight_tensor.data)
        else:
            in_features = 'unknown' # 提供一个默认值，增加容错性
            
        return f"{'ReLU' if self.nonlin else 'JOKER'}Linear({in_features})"

    