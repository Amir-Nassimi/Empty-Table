#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from numpy.linalg import norm as Norm
from singleton_decorator import singleton


import os, sys
from pathlib import Path

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[1]))
from utils.camera_ver_2 import Camera
from utils.detection_ver_2 import Detection


# In[2]:


desired_fps = 30


# In[3]:


class Employee:
    def __init__(self, emp_id, bbx):
        self._bbx = bbx
        self._emp_id = emp_id
        
        self._working_time = pd.Timedelta("0 day 00:00:00.00")
        self._chair_not_detected = pd.Timedelta("0 day 00:00:00.00")
        self._person_not_working = pd.Timedelta("0 day 00:00:00.00")
    
    def Update_Time(self, flag, desired_fps=30):
        if flag == 'working':
            self._working_time += pd.Timedelta(seconds=1/desired_fps)
        elif flag == 'not working':
            self._person_not_working += pd.Timedelta(seconds=1/desired_fps)
        else:
            self.__chair_not_detected += pd.Timedelta(seconds=1/desired_fps)
            
    @property
    def bbx(self):
        return self._bbx
    
    @property
    def emp_id(self):
        return self._emp_id
    
    @property
    def working_time(self):
        return self._working_time
    
    @property
    def chair_not_detected(self):
        return self._chair_not_detected
    
    @property
    def person_not_working(self):
        return self._person_not_working


# In[4]:


class Execution:
    def __init__(self, input_dict):
        self.employee = []
        self._detector = Detection()
        self._cam = Camera(input_dict['camera'], desired_fps=desired_fps)
    
    @staticmethod
    def Cropped_Img(frame,x,y,w,h,extension_percentage=0.15):
        image = frame.copy()
        
        height_extension = int(h * extension_percentage)
        width_extension = int(w * extension_percentage * 0.009)  # Smaller extension for width

        new_x = max(0, x - width_extension)
        new_y = max(0, y - height_extension)
        new_w = min(image.shape[1], w + 2 * width_extension)
        new_h = min(image.shape[0], h + height_extension)

        extended_image = image[new_y:new_h, new_x:new_w]
        return image[new_y:new_h, new_x:new_w], [new_x,new_y,new_w,new_h] 
        
    def Run(self):
        self._cam.run()
        
        counter = 0
        while not self._cam.exit_signal.is_set():
            frame = self._cam.frames_queue.get()
            
            detected_list = self._detector.Object_Detection(frame, labels=['chair'])
            
            target_list, chairs = [], []
            for indx_1, chair in enumerate(detected_list):
                img = chair[0]
                for indx_2, (_, coord) in enumerate(chair[1]):
                    new_img, new_coord = self.Cropped_Img(img,*coord)
                    target_list.append(new_img)
                    chairs.append((new_img, new_coord))
                    
                    if counter == 0:
                        self.employee.append(Employee(indx_1+indx_2, new_coord))
            
            persons = self._detector.Object_Detection(target_list, labels=['person'])
            
            for i in range(len(chairs)):
                try:
                    if persons[i][1]: self.employee[i].Update_Time('working')
                    else: self.employee[i].Update_Time('not working')
                except IndexError: self.employee[i].Update_Time('chair not detected')
            
            
            for employee in self.employee:
                print(f'employee : {employee.emp_id}\n\t Working: {employee.working_time}' +
                      f'\n\t Not Working: {employee.person_not_working}'+
                      f'\n\t Chair not detected: {employee.chair_not_detected}'
                     )
            
            counter += 1
        self._cam.stop_threads()


# In[5]:


x = {
    "camera": "rtsp://rtsp:Ashkan123@172.16.60.123/Streaming/Channels/401/",
}


# In[6]:


execution = Execution(x)
execution.Run()


# In[ ]:




