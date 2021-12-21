import os
import numpy as np
import tensorflow as tf
from Converter import Converter
import Layer_Models.Conv2D_1x1

class BenchMarkProblem:
    def _prepareExampleInputs(self):
        for x in range(self.size):
            self.inputs.append(self.model.getRandomInput())

    def __init__(self) -> None:
        self.size=100
        self.inputs=[]
        self._prepareExampleInputs()
    
    def run(self):
        return


class BenchMarkProblem1(BenchMarkProblem):
    def __init__(self) -> None:
        self.model=Layer_Models.Conv2D_1x1.Conv2D_1x1()
        self.converter=Converter(self.model)
        self.converter.convert(self.converter.targets.neural_stick_2)
        super().__init__()

    def run(self):
        
        return super().run()
        

    

