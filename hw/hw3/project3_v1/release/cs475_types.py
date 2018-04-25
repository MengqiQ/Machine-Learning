from abc import ABCMeta, abstractmethod
from collections import defaultdict

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        # TODO
        self.label = label
        
    def __str__(self):
        # TODO
        return str(self.label)



class FeatureVector:
    def __init__(self):
        # TODO
        self.dict = defaultdict(lambda: 0,0)
        pass
        
    def add(self, index, value):
        # TODO
        self.dict[index] = value
        pass
        
    def get(self, index):
        # TODO
        return self.dict[index]
        

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

       
