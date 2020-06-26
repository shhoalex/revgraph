class TensorMagic(object):
    """
    A class for implementing all the magic methods without the problem of
    circular imports.
    """
    def __add__(self, other):
        from revgraph.core.functions.operations.math.add import Add
        return Add(self, other)

    def __radd__(self, other):
        from revgraph.core.functions.operations.math.add import Add
        return Add(other, self)

    def __sub__(self, other):
        from revgraph.core.functions.operations.math.sub import Sub
        return Sub(self, other)

    def __rsub__(self, other):
        from revgraph.core.functions.operations.math.sub import Sub
        return Sub(other, self)

    def __mul__(self, other):
        from revgraph.core.functions.operations.math.mul import Mul
        return Mul(self, other)

    def __rmul__(self, other):
        from revgraph.core.functions.operations.math.mul import Mul
        return Mul(other, self)

    def __truediv__(self, other):
        from revgraph.core.functions.operations.math.truediv import TrueDiv
        return TrueDiv(self, other)

    def __rtruediv__(self, other):
        from revgraph.core.functions.operations.math.truediv import TrueDiv
        return TrueDiv(other, self)

    def __floordiv__(self, other):
        from revgraph.core.functions.operations.math.floordiv import FloorDiv
        return FloorDiv(self, other)

    def __rfloordiv__(self, other):
        from revgraph.core.functions.operations.math.floordiv import FloorDiv
        return FloorDiv(other, self)

    def __pow__(self, other):
        from revgraph.core.functions.operations.math.pow import Pow
        return Pow(self, other)

    def __rpow__(self, other):
        from revgraph.core.functions.operations.math.pow import Pow
        return Pow(other, self)

    def __matmul__(self, other):
        from revgraph.core.functions.operations.math.matmul import MatMul
        return MatMul(self, other)

    def __rmatmul__(self, other):
        from revgraph.core.functions.operations.math.matmul import MatMul
        return MatMul(other, self)

    def matmul(self, other):
        from revgraph.core.functions.operations.math.matmul import MatMul
        return MatMul(self, other)

    def dot(self, other):
        from revgraph.core.functions.operations.math.matmul import MatMul
        return MatMul(self, other)

    def __neg__(self):
        from revgraph.core.functions.operations.math.neg import Neg
        return Neg(self)

    def __call__(self, **kwargs):
        from revgraph.core.runner import run
        return run(self, kwargs)

    def sum(self, axis=None, keepdims=False):
        from revgraph.core.functions.operations.math.sum import Sum
        return Sum(self, axis=axis, keepdims=keepdims)

    def __len__(self):
        """
        Potentially buggy.
        """
        from revgraph.core.functions.miscellaneous import len
        return len(self)

    def __gt__(self, other):
        from revgraph.core.functions.miscellaneous import greater
        return greater(self, other)

    def __ge__(self, other):
        from revgraph.core.functions.miscellaneous import greater_equal
        return greater_equal(self, other)

    def __lt__(self, other):
        from revgraph.core.functions.miscellaneous import less
        return less(self, other)

    def __le__(self, other):
        from revgraph.core.functions.miscellaneous import less_equal
        return less_equal(self, other)

    def __eq__(self, other):
        from revgraph.core.functions.miscellaneous import equal
        return equal(self, other)

    def __hash__(self):
        return id(self)

    def __ne__(self, other):
        from revgraph.core.functions.miscellaneous import not_equal
        return not_equal(self, other)

    def all(self):
        from revgraph.core.functions.miscellaneous import all
        return all(self)

    def any(self):
        from revgraph.core.functions.miscellaneous import any
        return any(self)
