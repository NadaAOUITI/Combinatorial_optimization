from abc import ABC, abstractmethod

class Metaheuristic(ABC):
    @abstractmethod
    def optimize(self, instance):
        pass

class ExactMethod(ABC):
    @abstractmethod
    def optimize(self, instance):
        pass
