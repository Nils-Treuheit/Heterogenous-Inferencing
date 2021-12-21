import os
import enum

#Die Klasse soll das Model speichern und in die richtige Form bringen/ optimizer Aufrufen... 
class Converter:
    def __init__(self,model,path_input='Layer_Models',path_output='Converted_Models'):
        if os.uname().sysname=="Linux":
            self.openvino_location='/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py'
        else:
            print("Ich weiss nicht, wo die openVINO-Location bei Windows ist.")
        self.model=model
        self.source=path_input
        self.result=path_output
        self.targets=enum.Enum('TARGET','neural_stick_2 google_coral')

    def convert(self,target):
        if both_exist: #Falls beide existieren, überprüfen wir, ob die Quell-Datei jünger als unser konvertiertes Modell ist 
            if os.path.getmtime(os.path.join('.',self.source,self.model.model_Name))<os.path.join('.',self.result,self.model.model_Name):
                return

        if target==self.targets.neural_stick_2:
            self.convert_openvino()
        elif target==self.targets.google_coral:
            self.convert_coral()

    def convert_openvino(self):
        #command line options:
        # --saved_model_dir ./Layer_Models/Conv2D_1x1_Layer
        # --output_dir ./Converted_Models/Conv2D_1x1_Layer
        # --input_shape [1,100,100,1] -n model --data_type=FP16
        model_location=' --saved_model_dir \"'+os.path.join('.',self.source,self.model.model_Name)+"\""
        converted_model_location=' --output_dir \"'+os.path.join('.',self.result,self.model.model_Name)+"\""
        command='python3 ' +self.openvino_location
        command+=model_location+converted_model_location
        command+=' -n model --data_type=FP16 '
        command+=' --input_shape '+str(self.model.input_shape).replace(" ","")
        print("Executing: "+command)
        os.system(command)

