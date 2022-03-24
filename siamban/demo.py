'''
    Make some modification on Siamban
    functions:
        draw: add mouse click to choose the tracking objects
        LoadRealsense2: use realsense2 sdk to capture picture from D435i/D435
        get_depth: use depth img and camera params to compute the depth of object

    All the modification is mark as TODO.

'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from dis import dis

import os
import argparse
import sys

import cv2
from cv2 import data
from matplotlib import scale
from matplotlib.pyplot import box
import torch
import numpy as np
from glob import glob
import pyrealsense2 as rs
import rospy
from track_pkg.msg import Target

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--save', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

ix,iy = -1,-1
bx,by = -1,-1
flag = 0
i = 1
init = 0
anchor = (0,0,0,0)

#TODO
rospy.init_node("getTarget")
tarpub = rospy.Publisher("target",Target,queue_size=1)

#TODO
def draw(event,x,y,flags,param):
    global ix,iy,bx,by,flag,i,anchor
    if event == cv2.EVENT_LBUTTONDOWN:
        flag = 1
        ix, iy = x,y
        bx,by = -1,-1
    elif event == cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        if i ==1:
            i = i+1
        if i >1:
            pass
        bx, by = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        bx, by = x, y
        anchor = [ix, iy, bx-ix, by-iy]
        flag = 0


# def draw2(event,x,y,flags,param):
#     global ix,iy,bx,by,flag,anchor
#     if event == cv2.EVENT_LBUTTONDOWN:
#         flag = 1
#         ix, iy = x,y
#     elif event == cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
#         ix,iy = min(ix,x),min(iy,y)
#         anchor= [ix,iy,abs(x-ix),abs(y-iy)]
#     elif event == cv2.EVENT_LBUTTONUP:
#         bx, by = x, y
#         anchor = [ix, iy, bx-ix, by-iy]
#         flag = 0


fx = 609.2713012695312
fy = 608.010498046875
cx = 316.67022705078125
cy = 244.8178253173828
imgscale = 1000

#TODO
def get_depth(depth_image,bbox):
    x,y,w,h = bbox
    global imgscale,fx,fy,cx,cy
    import math
    depth = []
    depth.append(depth_image[y+math.ceil(h/3.0)][x+math.ceil(w/3.0)]/imgscale) 
    depth.append(depth_image[y+math.ceil(h*2.0/3.0)][x+math.ceil(w/3.0)]/imgscale) 
    depth.append(depth_image[y+math.ceil(h/3.0)][x+math.ceil(w*2.0/3.0)]/imgscale)
    depth.append(depth_image[y+math.ceil(h*2.0/3.0)][x+math.ceil(w*2.0/3.0)]/imgscale) 
    depth.append(depth_image[y+math.ceil(h/2.0)][x+math.ceil(w/2.0)]/imgscale)

    print(depth)

    num = 5
    dist_x = dist_y = 0
    for d in depth:
        if d > 0.0 and d <10.0:
            dist_x+=d
        else:
             num -= 1
    
    if(num):
        dist_x = dist_x/num

    list_y =[]

    list_y.append((x+math.ceil(w/3.0) - cx)*(depth[0]+depth[1])/(2.0*fx))
    list_y.append((x+math.ceil(w/3.0) - cx)*(depth[2]+depth[3])/(2.0*fx))
    list_y.append((x+math.ceil(w/3.0) - cx)*depth[4]/fx)

    print(list_y)

    num = 3
    for y in list_y:
        if y > -10.0 and y < 10.0:
            dist_y += y
        else:
            num -= 1
    if(num):
        dist_y/num

    return dist_x,dist_y
    
    

#TODO
def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(4)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4') or \
        video_name.endswith('mov'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


class LoadRealsense2:

    def __init__(self,px=640,py=480,fps=30):
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color,px,py,rs.format.bgr8,fps)
        self.config.enable_stream(rs.stream.depth,px,py,rs.format.z16,fps)
        self.profile = self.pipe.start(self.config)        
        self.align = rs.align(rs.stream.color)
    
    def __iter__(self):
        return self

    def __next__(self):
        if cv2.waitKey(1) == 'q':
            self.pipe.stop()
            cv2.destroyAllWindows()
            raise StopIteration

        frames= self.pipe.wait_for_frames()
        color = np.asanyarray((frames.get_color_frame()).get_data())
        aligned_depth_frames = self.align.process(frames)
        aligned_depth = np.asanyarray((aligned_depth_frames.get_depth_frame()).get_data())
        
        return color,aligned_depth

def main():
    # load config
    global ix,iy,bx,by,flag,i,anchor,init
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()
    tracker = build_tracker(model)
    
    video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setMouseCallback(video_name,draw)#回调鼠标
    
    # dataset = LoadRealsense2()
    # for color,depth in dataset:
    #     print(type(color))
    #     cv2.imshow('dsa',color)
    #     cv2.waitKey(10)
    #     return

    dataset = LoadRealsense2()

    for frame,depth in dataset:
        # print(type(frame)) #
        
        # if anchor[2]<=0 or anchor[3]<=0:
        #     cv2.imshow(video_name, frame)
        #     cv2.waitKey(10)
        #     continue

        if flag == 0 and init ==0 and anchor[2] != 0 and anchor[3] != 0:
            tracker.init(frame, anchor)
            init = 1
        elif flag == 0 and init ==1 and anchor[2] != 0 and anchor[3] != 0:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
                target = Target()
                target.x,target.y = get_depth(depth,bbox)
                tarpub.publish(target)
        elif flag == 1:
            print(ix,iy,bx,by)
            cv2.rectangle(frame, (ix, iy), (bx, by), (0, 0, 255), 3)
            init = 0
            tarpub.publish(Target(0.0,0.0))


        cv2.imshow(video_name, frame)
        cv2.waitKey(10)


if __name__ == '__main__':
    main()
