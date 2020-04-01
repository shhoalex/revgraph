from revgraph.core.functions.decorators import no_grad


@no_grad
def len(obj):
    return obj.__len__()


@no_grad
def greater_than(a,b):
    return a>b


@no_grad
def less_than(a,b):
    return a<b


@no_grad
def greater_than_or_equal(a,b):
    return a>=b


@no_grad
def less_than_or_equal(a,b):
    return a<=b


@no_grad
def equal(a,b):
    return a==b


@no_grad
def not_equal(a,b):
    return a!=b


@no_grad
def all(a, *args, **kwargs):
    return a.all(*args, **kwargs)


@no_grad
def any(a, *args, **kwargs):
    return a.any(*args, **kwargs)


@no_grad
def argmax(a, *args, **kwargs):
    return a.argmax(*args, **kwargs)


@no_grad
def argmin(a, *args, **kwargs):
    return a.argmin(*args, **kwargs)
