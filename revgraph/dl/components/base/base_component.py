from abc import abstractmethod, ABC
from typing import Any, List, Tuple

import revgraph.core as rc


class BaseComponent(ABC):
    def __init__(self,
                 *args: Any,
                 **kwargs: Any):
        self.verify(self.specs(*args, **kwargs))

    @staticmethod
    def verify(predicates: List[Tuple[bool, str]]) -> None:
        for valid, err_msg in predicates:
            if not valid:
                raise ValueError(err_msg)

    @abstractmethod
    def specs(self,
              *args: Any,
              **kwargs: Any) -> List[Tuple[bool, str]]:
        pass

    @abstractmethod
    def instantiate(self,
                    *args: Any,
                    **kwargs: Any) -> None:
        pass

    @abstractmethod
    def build_graph(self) -> rc.tensor:
        pass
