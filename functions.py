import numpy as np

class NodeFunction:
    @staticmethod
    def forward():
        pass

    @staticmethod
    def backward():
        pass

class Sum(NodeFunction):
    @staticmethod
    def forward(node, previous):
        x, y = previous
        node.value =  x.value + y.value

    @staticmethod
    def backward(node, previous):
        x, y = previous
        x.grad += node.grad
        y.grad += node.grad

class Mul(NodeFunction):
    @staticmethod
    def forward(node, previous):
        x, y = previous
        node.value = x.value * y.value
    @staticmethod
    def backward(node, previous):
        x, y = previous
        x.grad += y.value * node.grad
        y.grad += x.value * node.grad

class Relu(NodeFunction):
    @staticmethod
    def forward(node, previous):
        x, = previous
        node.value = x.value if x.value > 0 else 0
    @staticmethod
    def backward(node, previous):
        x, = previous
        x.grad += node.grad * (node.value > 0)

class Exp(NodeFunction):
    @staticmethod
    def forward(node, previous):
        x, = previous
        node.value = np.exp(x.value)
    @staticmethod
    def backward(node, previous):
        x, = previous
        x.grad += node.grad * np.exp(x.value)

class Square(NodeFunction):
    @staticmethod
    def forward(node, previous):
        x, = previous
        node.value = x.value**2
    @staticmethod
    def backward(node, previous):
        x, = previous
        x.grad += node.grad * 2 * x.value

class Log(NodeFunction):
    @staticmethod
    def forward(node, previous):
        x, = previous
        node.value = np.log(x.value)
    @staticmethod
    def backward(node, previous):
        x, = previous
        x.grad += node.grad * (1/x.value)

class Power(NodeFunction):
    def __init__(self, power):
        self.power = power

    def forward(self, node, previous):
        x, = previous
        node.value = x.value**self.power
    def backward(self, node, previous):
        x, = previous
        x.grad += node.grad * self.power * x.value**(self.power-1)

class Constant(NodeFunction):
    @staticmethod
    def forward(node, previous):
        return
    @staticmethod
    def backward(node, previous):
        return