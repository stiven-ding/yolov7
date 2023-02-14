import argparse
import time
from pathlib import Path

import cv2
import pyrealsense2 as rs
import numpy as np

def detect(save_img=False):

    # init realsense
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    sensor_dep = profile.get_device().first_depth_sensor()
    sensor_dep.set_option(rs.option.min_distance, 30)
    sensor_dep.set_option(rs.option.enable_max_usable_range, 0)
    sensor_dep.set_option(rs.option.laser_power, 40)

    align_to = rs.stream.color
    align = rs.align(align_to)

    while(True):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        # filtering
        #spatial = rs.spatial_filter(smooth_alpha=0.25,smooth_delta=20,magnitude=2,hole_fill=3)
        #depth_frame = spatial.process(depth_frame)
        hole_filling = rs.hole_filling_filter()
        depth_frame = hole_filling.process(depth_frame)
        #temporal_filter = rs.temporal_filter()
        #depth_frame = temporal_filter.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        #depth_image = cv2.morphologyEx(depth_image,cv2.MORPH_OPEN,kernel,iterations=1)
        depth_image = cv2.medianBlur(depth_image,5)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        im0 = color_image.copy()

        depth_scale = sensor_dep.get_depth_scale()

        # range selection
        depth_l1 = cv2.inRange(depth_image,0.1/depth_scale, 0.5/depth_scale)
        depth_l1_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_l1, alpha=0.2), cv2.COLORMAP_JET)

        depth_l2 = cv2.inRange(depth_image,0.5/depth_scale, 1.0/depth_scale)
        depth_l2_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_l2, alpha=0.2), cv2.COLORMAP_JET)

        contours_l1, _ = cv2.findContours(depth_l1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours_l2, _ = cv2.findContours(depth_l2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(depth_l1_colormap, contours_l1, -1, (0,255,0), 2)
        cv2.drawContours(depth_l2_colormap, contours_l2, -1, (0,255,0), 2)

        # Stream results
        cv2.imshow("Recognition result", im0)
        cv2.imshow("L1 result depth",cv2.cvtColor(depth_l1_colormap, cv2.COLOR_BGR2RGB))
        cv2.imshow("L2 result depth",cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.torchscript.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    detect()
