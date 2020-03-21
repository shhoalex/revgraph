from abc import abstractmethod, ABC


class Computation(ABC):
    @abstractmethod
    def forward(self):
        pass
