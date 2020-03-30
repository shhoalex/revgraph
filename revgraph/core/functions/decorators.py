from revgraph.core.functions.base.no_grad_function import NoGradFunction


def no_grad(f):
    class Function(NoGradFunction):
        def call(*args, **kwargs):
            return f(*args, **kwargs)
    return Function
