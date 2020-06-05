from revgraph.core.functions.base.no_grad_function import NoGradFunction


def no_grad(f):
    class NoGradDeriv(NoGradFunction):
        __name__ = f.__name__
        __qualname__ = f.__qualname__

        def call(*args, **kwargs):
            return f(*args, **kwargs)
    return NoGradDeriv
