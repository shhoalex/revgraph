class ComputationMagic(object):
    def __add__(self, other):
        from revgraph.core.functions.operations.add import Add
        return Add(self, other)

    def __radd__(self, other):
        from revgraph.core.functions.operations.add import Add
        return Add(other, self)

    def __sub__(self, other):
        from revgraph.core.functions.operations.sub import Sub
        return Sub(self, other)

    def __rsub__(self, other):
        from revgraph.core.functions.operations.sub import Sub
        return Sub(other, self)

    def __mul__(self, other):
        from revgraph.core.functions.operations.mul import Mul
        return Mul(self, other)

    def __rmul__(self, other):
        from revgraph.core.functions.operations.mul import Mul
        return Mul(other, self)

    def __truediv__(self, other):
        from revgraph.core.functions.operations.truediv import TrueDiv
        return TrueDiv(self, other)

    def __rtruediv__(self, other):
        from revgraph.core.functions.operations.truediv import TrueDiv
        return TrueDiv(other, self)

    def __floordiv__(self, other):
        from revgraph.core.functions.operations.floordiv import FloorDiv
        return FloorDiv(self, other)

    def __rfloordiv__(self, other):
        from revgraph.core.functions.operations.floordiv import FloorDiv
        return FloorDiv(other, self)

    def __pow__(self, other):
        from revgraph.core.functions.operations.pow import Pow
        return Pow(self, other)

    def __rpow__(self, other):
        from revgraph.core.functions.operations.pow import Pow
        return Pow(other, self)

    def __matmul__(self, other):
        from revgraph.core.functions.operations.matmul import MatMul
        return MatMul(self, other)

    def __rmatmul__(self, other):
        from revgraph.core.functions.operations.matmul import MatMul
        return MatMul(other, self)

    def matmul(self, other):
        from revgraph.core.functions.operations.matmul import MatMul
        return MatMul(self, other)

    def dot(self, other):
        from revgraph.core.functions.operations.matmul import MatMul
        return MatMul(self, other)

    def __neg__(self):
        from revgraph.core.functions.operations.neg import Neg
        return Neg(self)

    def __call__(self, feed_dict=None, *args, **kwargs):
        from revgraph.core.runner import run
        [result] = run(self, feed_dict, [self])
        return result

    def sum(self, axis=None):
        from revgraph.core.functions.math.sum import Sum
        return Sum(self, axis=axis)

    def __len__(self):
        from revgraph.core.functions.miscellaneous import len
        return len(self)

    def __gt__(self, other):
        from revgraph.core.functions.miscellaneous import greater_than
        return greater_than(self, other)

    def __ge__(self, other):
        from revgraph.core.functions.miscellaneous import greater_than_or_equal
        return greater_than_or_equal(self, other)

    def __lt__(self, other):
        from revgraph.core.functions.miscellaneous import less_than
        return less_than(self, other)

    def __le__(self, other):
        from revgraph.core.functions.miscellaneous import less_than_or_equal
        return less_than_or_equal(self, other)

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
