from abc import abstractmethod, ABC
from typing import Any, Dict, List, Tuple


class BaseComponent(ABC):
    def __init__(self,
                 *args: Any,
                 **kwargs: Any):
        self.verify(*args, **kwargs)

    @staticmethod
    def verify(predicates: List[Tuple[bool, str]]) -> None:
        for valid, err_msg in predicates:
            if not valid:
                raise ValueError(err_msg)

    @abstractmethod
    def specs(self, *args, **kwargs) -> List[Tuple[bool, str]]:
        pass

    @abstractmethod
    def build(self) -> Dict[str, Any]:
        pass
