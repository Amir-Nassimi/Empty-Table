# In[ ]:

import numpy as np
from ultralytics import YOLO
from singleton_decorator import singleton
from torch import no_grad as Torch_No_Grad

import os
from pathlib import Path
# In[ ]:


@singleton
class Detection:
    def __init__(self, thresh = 45, model_path='yolov8x.pt'):
        self._model = self.Load_Model(model_path)
        self.Warm_Up()
        self.thresh = thresh
        
    def Warm_Up(self):
        print("Warming Up Model!")
        with Torch_No_Grad(): self._model.predict(np.zeros((640,640,3)))
        print("Model Warmed Up!")
    
    @staticmethod
    def Load_Model(pth):
        model = YOLO(f'{os.path.abspath(Path(__file__).resolve().parents[2])}/Models/{pth}')
        print("Model Loaded Successfuly!!")
        return model
    
    def Object_Detection(self, img, labels=['chair'], verbose=False):
        cordinations = []
        
        with Torch_No_Grad(): results = self._model.predict(img)
        
        if len(results[0]) == 0: pass
        else:
            if type(img) == list: img_len = len(img)
            else: img_len = 1
            
            for i in range(img_len):
                temp = []
                
                for result in results[i]:
                    for box in result.boxes:
                        predict = result.names[box.cls.item()]
                        probe = box.conf.item() * 100
                        bounding_boxes = box.xyxy.tolist()[0]

                        if verbose: print(f'{predict} - {probe}')

                        if predict in labels:

                            if probe > self.thresh:
                                x = int(round(bounding_boxes[0]))
                                y = int(round(bounding_boxes[1]))
                                w = int(round(bounding_boxes[2]))
                                h = int(round(bounding_boxes[3]))

                                temp.append((predict,[x,y,w,h]))

                if type(img) == list:
                    cordinations.append((img[i],temp))
                    
                else: cordinations.append((img,temp))

        return cordinations