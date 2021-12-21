import os


class Our_Model:
    def __init__(self):
        self.model_Name=Our_Model
        self.input_shape=[]

    def getModel(self): #Aufrufen, um Modell zu erstellen
        pass

    def saveModel(self): #Aufrufen, um Modell zu Speichern (Unter Layer_Models)
        model=self.model
        #model.fit()
        #model.
        model.save(os.path.join(".","Layer_Models",self.model_Name))

    def getRandomInput(self):
        pass