import numpy as np

class QuantileParam:
    epsilon = 0.1

class FeatureData:
    def __init__(self, name, dataVector) -> None:
        self.name = name
        self.data = dataVector

class QuantiledFeature(FeatureData):
  
    def __init__(self, name, dataVector) -> None:
        super().__init__(name, dataVector)
        self.splittingMatrix = QuantiledFeature.quantile()
    
    def quantile(data: FeatureData, param: QuantileParam):
        splittingMatrix = []
        return splittingMatrix

class DataBase:
    def __init__(self) -> None:
        self.data = []

    def appendFeature(featureData: FeatureData):
        pass

