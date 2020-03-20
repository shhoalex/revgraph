from abc import abstractmethod, ABC
from typing import Dict


class Computation(ABC):
    @abstractmethod
    def evaluate(self, feed_dict: Dict[str, 'Computation']):
        pass
