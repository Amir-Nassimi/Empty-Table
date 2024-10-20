import pandas as pd
from numpy.linalg import norm as Norm
from singleton_decorator import singleton

import os, sys
from pathlib import Path

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[1]))
from utils.camera_ver_1_2 import Camera
from utils.detection_ver_1_2 import Detection

desired_fps = 30

#Add to DB
class Employee:
    def __init__(self, emp_id, bbx):
        self._bbx = bbx
        self._emp_id = emp_id
        
        self._working_time = pd.Timedelta("0 day 00:00:00.00")
        self._person_not_working = pd.Timedelta("0 day 00:00:00.00")
        self._chair_not_detected = pd.Timedelta("0 day 00:00:00.00")
        self._person_at_table_no_sitting = pd.Timedelta("0 day 00:00:00.00")
    
    def Update_Time(self, flag, desired_fps=30):
        if flag == 'working':
            self._working_time += pd.Timedelta(seconds=1/desired_fps)
        elif flag == 'not_working':
            self._person_not_working += pd.Timedelta(seconds=1/desired_fps)
        elif flag == 'chair_not_detected':
            self._chair_not_detected += pd.Timedelta(seconds=1/desired_fps)
        else:
            self._person_at_table_no_sitting += pd.Timedelta(seconds=1/desired_fps)

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
    def person_not_working(self):
        return self._person_not_working
    
    @property
    def chair_not_detected(self):
        return self._chair_not_detected
    
    @property
    def person_at_table_no_sitting(self):
        return self._person_at_table_no_sitting


@singleton
class Empty_Table:
    
    @staticmethod
    def _Overlap_Percent(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0

        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)
        iou = iou * 100

        return iou
    
    @staticmethod
    def _Area_Calc(bbx):
        area = (bbx[2] * bbx[3]) / 100
        if area >= 2000:
            dist_tresh = ((int(bbx[0] + bbx[2]) / int(bbx[1] + bbx[3])) * 100 + 10 * 2)
            person_confidence_tresh = 30

        elif 1300 <= area < 2000:
            dist_tresh = ((int(bbx[0] + bbx[2]) / int(bbx[1] + bbx[3])) * 100 + 10 * 3.25)
            person_confidence_tresh = 30

        elif 800 <= area < 1300:
            dist_tresh = ((int(bbx[0] + bbx[2]) / int(bbx[1] + bbx[3])) * 100 + 10 * (-3.25))
            person_confidence_tresh = 30

        elif 150 <= area < 800:
            dist_tresh = ((int(bbx[0] + bbx[2]) / int(bbx[1] + bbx[3])) * 100 + 10 * 6.15)
            person_confidence_tresh = 30

        else:
            dist_tresh = ((int(bbx[0] + bbx[2]) / int(bbx[1] + bbx[3])) *  100 + 10 * 3)
            person_confidence_tresh = 40

        return dist_tresh, person_confidence_tresh
    
    def Calculation(self, chairs_list, persons_list, employee, frame):               
        dist_tresh,_ = self._Area_Calc(employee.bbx)
        frame_cropped = frame[employee.bbx[1]:employee.bbx[1] + employee.bbx[3], employee.bbx[0]:employee.bbx[0] + employee.bbx[2]]
        
        chair_list, person_list, overlap_list = [], [], []
        
        if persons_list:
            if chairs_list:
                for _,person in persons_list:
                    for _,chair in chairs_list:
                        coverage = self._Overlap_Percent(person, chair)    
                        overlap_list.append(coverage) 
                        person_list.append(person)
                        chair_list.append(chair)
                        
                if max(overlap_list) < 15: employee.Update_Time('not_working')
                else:
                    index = overlap_list.index(max(overlap_list))
                    person = person_list[index]
                    chair = chair_list[index]

                    if person[1] > chair[1]: employee.Update_Time('working')
                    else:
                        dist_y =Norm(person[1] - chair[1])
                        if dist_y >= dist_tresh: employee.Update_Time('person_at_table_not_sitting')
                        else: employee.Update_Time('working')
                            
            else: employee.Update_Time('chair_not_detected')
        else: employee.Update_Time('not_working')
                           

class Execution:
    def __init__(self, input_dict):
        self._detector = Detection()
        self._empty_table = Empty_Table()
        self._cam = Camera(input_dict['camera'], desired_fps=desired_fps)
        
        self._employees = []
        for key,value in input_dict['table'].items():
            self._employees.append(Employee(key,value))
    
    @staticmethod
    def Divide_Array(array):
        is_person = lambda label: label == 'person'
        is_chair = lambda label: label == 'chair'

        person_list = [(image, bbox) for image, labels in array for label_type, bbox in labels if is_person(label_type)]
        chair_list = [(image, bbox) for image, labels in array for label_type, bbox in labels if is_chair(label_type)]

        return person_list, chair_list

    def Run(self):
        self._cam.run()
        
        while not self._cam.exit_signal.is_set():
            frame = self._cam.frames_queue.get()
            
            detected_list = self._detector.Object_Detection(frame, labels=['chair','person'])
            persons, chairs = self.Divide_Array(detected_list)
            del detected_list
            
            for employee in self._employees:
                self._empty_table.Calculation(chairs, persons, employee, frame)
                # Result
                print(f'employee : {employee.emp_id}\n\t Working: {employee.working_time}' +
                      f'\n\t Not Working: {employee.person_not_working}' + 
                      f'\n\t Person Standing: {employee.person_at_table_no_sitting}' + 
                      f'\n\t No Chair Detected: {employee.chair_not_detected}'
                     )
            
        self._cam.stop_threads()

x = {
    "camera": "rtsp://rtsp:Ashkan123@172.16.60.123/Streaming/Channels/101/",
    "table":
        {
            "512215":[125,45,320,550],
            "513125":[136,105,451,215],
            "512320" : [159-20, 286-6, 180+15, 306+40]
      }
}

execution = Execution(x)
execution.Run()