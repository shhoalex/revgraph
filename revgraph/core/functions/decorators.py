from revgraph.core.functions.base.no_grad_function import NoGradFunction


def no_grad(f):
    """
    Creates a NoGradFunction that dynamically evaluates function f at runtime.
    """
    def callback(*args, **kwargs):
        g = NoGradFunction(*args, **kwargs)
        g.target_function = f
        return g
    return callback
