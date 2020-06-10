from typing import List, Tuple


def validate(*predicates: Tuple[bool, str]) -> None:
    for valid, err_msg in predicates:
        if not valid:
            raise ValueError(err_msg)
