# In[1]:


import torch
import numpy as np

from singleton_decorator import singleton


# In[2]:


@singleton
class Detection:
    def __init__(self):
        self._model = self._Load_Model()
        self._Warm_Up()
        
    def _Warm_Up(self):
        print("Warming Up Model!")
        with torch.no_grad(): self._model(np.zeros((640,640,3)))
        print("Model Warmed Up!")
    
    @staticmethod
    def _Load_Model():
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5x6.pt',device='cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Model Loaded Successfuly!!")
        return model
    
    def Object_Detection(self,img):
        chair_cordination = []
        person_cordination = []

        with torch.no_grad(): results = self._model(img)

        if len(results.xyxy[0]) == 0: pass
        else:        
            for indx,res in enumerate(results.xyxy[0]):
                predict = results.pandas().xyxy[0]['name'][indx]
                
                if predict == "chair" or predict == "person":
                    res = np.array(res.detach().cpu())
                    
                    if res[4]*100 > 45:
                        x = int(round(res[0]))
                        y = int(round(res[1]))
                        w = int(round(res[2]))
                        h = int(round(res[3]))
                        
                        if predict == "chair": chair_cordination.append((img[y:h, x:w], [x,y,w,h]))
                        else: person_cordination.append((img[y:h, x:w], [x,y,w,h]))
                        
                else: continue

        return chair_cordination,person_cordination

