# In[1]:


import cv2


# In[2]:

class Camera:
    def __init__(self, source, desired_fps=30):
        self.source = source
        self._fps = desired_fps
        self._frames_queue = []
        self._capture = cv2.VideoCapture(self.source)
        self._capture.set(cv2.CAP_PROP_FPS, self._fps)

    def capture_frame(self):
        ret, frame = self._capture.read()
        if ret:
            return frame
        else:
            return None

    def run(self):
        pass  # No need to start threads

    def stop(self):
        self._capture.release()

    @property
    def frames_queue(self):
        return self._frames_queue

