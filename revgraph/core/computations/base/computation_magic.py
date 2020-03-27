# computation_magic.py


class ComputationMagic(object):
    def __add__(self, other):
        from revgraph.core.computations.functions.operations.add import Add
        return Add(self, other)

    def __radd__(self, other):
        from revgraph.core.computations.functions.operations.add import Add
        return Add(other, self)

    def __sub__(self, other):
        from revgraph.core.computations.functions.operations.sub import Sub
        return Sub(self, other)

    def __rsub__(self, other):
        from revgraph.core.computations.functions.operations.sub import Sub
        return Sub(other, self)

    def __mul__(self, other):
        from revgraph.core.computations.functions.operations.mul import Mul
        return Mul(self, other)

    def __rmul__(self, other):
        from revgraph.core.computations.functions.operations.mul import Mul
        return Mul(other, self)

    def __truediv__(self, other):
        from revgraph.core.computations.functions.operations.truediv import TrueDiv
        return TrueDiv(self, other)

    def __rtruediv__(self, other):
        from revgraph.core.computations.functions.operations.truediv import TrueDiv
        return TrueDiv(other, self)

    def __floordiv__(self, other):
        from revgraph.core.computations.functions.operations.floordiv import FloorDiv
        return FloorDiv(self, other)

    def __rfloordiv__(self, other):
        from revgraph.core.computations.functions.operations.floordiv import FloorDiv
        return FloorDiv(other, self)

    def __pow__(self, other):
        from revgraph.core.computations.functions.operations.pow import Pow
        return Pow(self, other)

    def __rpow__(self, other):
        from revgraph.core.computations.functions.operations.pow import Pow
        return Pow(other, self)

    def __matmul__(self, other):
        from revgraph.core.computations.functions.operations.matmul import MatMul
        return MatMul(self, other)

    def __rmatmul__(self, other):
        from revgraph.core.computations.functions.operations.matmul import MatMul
        return MatMul(other, self)

    def matmul(self, other):
        from revgraph.core.computations.functions.operations.matmul import MatMul
        return MatMul(self, other)

    def dot(self, other):
        from revgraph.core.computations.functions.operations.matmul import MatMul
        return MatMul(self, other)

    def __neg__(self):
        from revgraph.core.computations.functions.operations.neg import Neg
        return Neg(self)

    def __call__(self, feed_dict=None, *args, **kwargs):
        from revgraph.core.computations.runner import Runner
        return Runner(node=self).run(feed_dict)
