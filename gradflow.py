import numpy as np
import functions

class Node:
    def __init__(self, value, previous=(), function=functions.Constant):
        self.previous = previous
        self.function = function
        self.value = value
        function.forward(self, previous)
        self._backward = lambda: function.backward(self, previous)
        self.grad = 0

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.previous:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        if self.grad != 0:
            for prev in self.previous:
                prev.zero_grad()
        self.grad = 0

    @staticmethod
    def relu(x):
        return Node(0, (x, ), functions.Relu)

    @staticmethod
    def square(x):
        return Node(0, (x, ), functions.Square)

    @staticmethod
    def exp(x):
        return Node(0, (x, ), functions.Exp)

    @staticmethod
    def log(x):
        return Node(0, (x, ), functions.Log)

    def __add__(self, other):
        if not isinstance(other, Node):
            other = Node(other)
        return Node(0, (self, other), functions.Sum)
    
    def __mul__(self, other):
        if not isinstance(other, Node):
            other = Node(other)
        return Node(0, (self, other), functions.Mul)

    def __pow__(self, m):
        assert isinstance(m, (int, float))
        return Node(0, (self, ), functions.Power(m))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self*other

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)
