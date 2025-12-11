from abc import ABC, abstractmethod
import numpy as np

#-----Agent abstract base class: Inheritance + Polymorphism-----
class Agent(ABC):
    """
    Agent base class(abstract class)
    *Every concrete Agent should inherit from this class and implement act() and reset()
    *Also realize polymorphism
    """
    def __init__(self, action: int, max_action: float):
        self.action=action
        self.max_action=max_action

    @abstractmethod
    def act(self, observation: np.ndarray)->np.ndarray:
        """
        Return action according to observation(need overwritten)
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        """
        Initialize when episode start
        """
        raise NotImplementedError