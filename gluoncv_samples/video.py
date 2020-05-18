import os
import sys
import time
import socket
import json
import cv2

import logging as log

class Video():
    """
    Class to make a video an iterator object, returning next frame on each iteration.
    """

    def __init__(self, video_file, as_file=False):
        self._cap = cv2.VideoCapture(video_file)
        self._cap.open(video_file)
        self._width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) #OpenCV 3+
        print(f"Total frames: {self._total_frames}")
        self._frame_nr = 0
        self._as_file = as_file
    
    def __iter__(self):
        return self

    def __next__(self):
        if not self._cap.isOpened:
            self._close_cap()
            raise StopIteration
        else:
            self._frame_nr += 1
            print(f"Frame {self._frame_nr} of {self._total_frames}")
            flag, raw_frame = self._cap.read()
            if not flag:
                self._close_cap()
                raise StopIteration

            if self._as_file:
                return self._save_image(raw_frame)
            else:
                return raw_frame
            
    def __len__(self):
        return self._total_frames

    def _close_cap(self):
        self._cap.release()
        cv2.destroyAllWindows()

    def _save_image(self, frame):
        img_path = f'tmp/img{self._frame_nr}.jpg'
        cv2.imwrite(img_path, frame)
        return img_path
