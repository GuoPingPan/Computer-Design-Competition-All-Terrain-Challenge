'''
    Detector

    author: guopingpan
    email: 731061720@qq.com
           or panguoping02@gmail.com

    brief: This project is to use ResNet to recognize Animals and broadcast the animal's name by playsound.
'''

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import paddlex as pdx
import numpy as np
import cv2
import glob
from playsound import playsound
import pyrealsense2 as rs



def detect():
    '''
        equipment: D435i
    '''

    # 1.use realsense sdk to process the cam video
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color,640,480,rs.format.rgb8,30)
    pipeline.start(config) 

    # 2.load the predictor
    predictor = pdx.deploy.Predictor('./assets',use_gpu=False)

    # 3.detecting
    while(cv2.waitKey(33) is not ord('q')):

        frameset = pipeline.wait_for_frames()
        img = np.asanyarray(frameset.get_color_frame().get_data())
        cv2.imshow('camera',img)
        
        result = predictor.predict(image=img)
        print(result)
        if(result[0]['category'] == 'chicken' and result[0]['score'] > 0.98):
            playsound('chicken.mp3')

        elif(result[0]['category'] == 'sheep' and result[0]['score'] > 0.98):
            playsound('sheep.mp3')

        elif(result[0]['category'] == 'squirrel' and result[0]['score'] > 0.98):
            playsound('squirrel.mp3')
        print('play sound')
        print('\n')


if __name__=='__main__':
    detect()