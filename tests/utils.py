import revgraph.core as rc


def match_structure(a: rc.tensor, b: rc.tensor) -> bool:
    """
    See whether 2 tensors are equal structurally, not by value

    Examples:
    >>> x = rc.variable([1])

    >>> a = rc.constant([0]) + rc.tanh(x) * rc.placeholder(name='x')
    >>> b = rc.constant([1]) + rc.tanh(x) * rc.placeholder(name='x')
    >>> match_structure(a, b)
    True

    >>> c = rc.constant(0) + rc.sinh(x) * rc.placeholder(name='x')
    False

    >>> d = rc.variable(0) + rc.tanh(x) * rc.placeholder(name='x')
    >>> match_structure(a, d)
    False

    >>> e = rc.variable([0]) + rc.tanh(x) * rc.placeholder(name='y')
    >>> match_structure(a, e)
    False

    >>> f = rc.constant([2]) + rc.tanh(rc.variable(2)) * rc.placeholder(name='x')
    >>> match_structure(a, f)
    True
    """
    if type(a) != type(b):
        print(f'{type(a)} != {type(b)}')
        return False
    if isinstance(a, rc.function_primitive):
        a: rc.function_primitive
        b: rc.function_primitive
        if len(a.args) != len(b.args):
            print(f'len({type(a)}) != len({type(b)})')
            return False
        for arg_a, arg_b in zip(a.args, b.args):
            if not match_structure(arg_a, arg_b):
                print('match_structure on args failed')
                return False
        return True
    elif isinstance(a, rc.placeholder) and isinstance(b, rc.placeholder):
        a: rc.placeholder
        b: rc.placeholder
        if a.name != b.name:
            print(f'{type(a)}{a.name} != {type(b)}{b.name}')
        return a.name == b.name
    return True
