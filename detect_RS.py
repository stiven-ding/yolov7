import argparse
from pickle import TRUE
import time
from pathlib import Path

import cv2

from numpy import random


import pyrealsense2 as rs
import numpy as np

ENABLE_YOLO_DETECT = False

if ENABLE_YOLO_DETECT:
    import torch
    import torch.backends.cudnn as cudnn
    from models.experimental import attempt_load
    from utils.datasets import LoadStreams, LoadImages
    from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
        scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
    from utils.plots import plot_one_box
    from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def sample_distance(depth_image, mid_x, mid_y):
    global depth_scale
    window = 2

    sample_depth = depth_image[mid_y-window:mid_y+window, mid_x-window:mid_x+window].astype(float)
    dist, _, _, _ = cv2.mean(sample_depth)
    dist = dist * depth_scale

    return dist

def detect(save_img=False):

    global device, model, half, depth_scale
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Initialize YOLO
    if ENABLE_YOLO_DETECT:

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        print('Loading model\n')
        model = attempt_load(weights, map_location=device)  # load FP32 model
        if trace:
            model = TracedModel(model, device, opt.img_size)

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if half:
            model.half()  # to FP16

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # Reset camera, bug workaround. Min distance invalid when used
    #print("RS reset start")
    #ctx = rs.context()
    #devices = ctx.query_devices()
    #for dev in devices:
    #    dev.hardware_reset()
    #print("RS reset done")

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    sensor_dep = profile.get_device().first_depth_sensor()
    sensor_dep.set_option(rs.option.min_distance, 100)
    #sensor_dep.set_option(rs.option.enable_max_usable_range, 1)
    sensor_dep.set_option(rs.option.laser_power, 100)
    sensor_dep.set_option(rs.option.receiver_gain, 18)
    sensor_dep.set_option(rs.option.confidence_threshold, 1)
    sensor_dep.set_option(rs.option.noise_filtering, 2)

    depth_scale = sensor_dep.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:
        #t0 = time.time()
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except:
                print('Cam recv no frames')
            else:
                break

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        # filtering
        #depth_to_disparity = rs.disparity_transform(True)
        #disparity_to_depth = rs.disparity_transform(False)
        spatial = rs.spatial_filter(smooth_alpha=1,smooth_delta=50,magnitude=5,hole_fill=3)
        hole_filling = rs.hole_filling_filter(1)

        #depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        #depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        img = np.asanyarray(color_frame.get_data())
        im0 = img.copy()

        depth_img = np.asanyarray(depth_frame.get_data())
        invalid = np.full((480,640),255, dtype=np.uint8)
        depth_img = np.where(depth_img[:,:] == [0,0], invalid, depth_img)

        #depth_img = cv2.bilateralFilter((depth_img/256.0).astype(np.uint8), 9, 75, 75)
        depth_img = cv2.medianBlur(depth_img,5)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

        # Depth range selection
        #depth_l1_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_l1, alpha=0.2), cv2.COLORMAP_JET)
        #depth_l2 = cv2.inRange(depth_image,0.5/depth_scale, 1.0/depth_scale)
        #depth_l2_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_l2, alpha=0.2), cv2.COLORMAP_JET)
        #contours_l2, _ = cv2.findContours(depth_l2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #

        # Process contours
        #edged = cv2.Canny(depth_img.astype(np.uint8), 50, 200)
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        #dilate = cv2.dilate(edged, kernel, iterations =1)
        #contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(depth_colormap, contours, -1, (0,255,0), 2)

        contours = []
        c_start, c_step, c_levels = 0.0, 0.5, 3
        for i in range(c_levels):
            depth_range = cv2.inRange(depth_img,c_start/depth_scale, (c_start+c_step)/depth_scale)
            contours_range, _ = cv2.findContours(depth_range,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for c in contours_range:
                contours.append(c)
            #cv2.imshow("Result depth" + str(c_start) + " " + str(c_start+c_step),cv2.cvtColor(depth_range, cv2.COLOR_BGR2RGB))
            c_start += c_step

        for c in contours:
            #cv2.convexHull(c)
            size = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)

            if w > 50 and h > 50 and w < 640 and h < 480:
                cv2.rectangle(depth_colormap, (x,y), (x+w,y+h), (0,200,100),2)
                M = cv2.moments(c)
                mid_x, mid_y = int(M['m10']/M['m00']), int(M['m01']/M['m00'])

                dist = sample_distance(depth_img, mid_x, mid_y)
                cv2.putText(depth_colormap, "dist: " + str(round(dist,2)) + "m", 
                (x, y-7),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,200,0),2)

        if ENABLE_YOLO_DETECT:

            # YOLO detect
            img = img[np.newaxis, :, :, :]        

            # Stack
            img = np.stack(img, 0)

            # Convert
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            img = np.ascontiguousarray(img)


            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            old_img_w = old_img_h = opt.img_size
            old_img_b = 1

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            #t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            #t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            #t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    #for c in det[:, -1].unique():
                        #n = (det[:, -1] == c).sum()  # detections per class
                        #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        obj_w = xyxy[2]-xyxy[0]
                        obj_h = xyxy[3]-xyxy[1]

                        if obj_w > 50 and obj_h > 50 and obj_w < 640 and obj_h < 480:
                            mid_x, mid_y = round(int(xyxy[0]+xyxy[2]) /2), round(int(xyxy[1] + xyxy[3])/2)

                            c = int(cls)  # integer class
                            dist = sample_distance(depth_img, mid_x, mid_y)
                            label = f'{names[c]} {dist:.2f}m'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                            plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)             

                # Print time (inference + NMS)
                #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        # Stream results
        cv2.imshow("YOLOv7 result", im0)
        #cv2.imshow("Depth L1 result",cv2.cvtColor(depth_l1_colormap, cv2.COLOR_BGR2RGB))
        cv2.imshow("L result depth",cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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

    if ENABLE_YOLO_DETECT:
        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov7.pt']:
                    detect()
                    strip_optimizer(opt.weights)
            else:
                detect()
    else:
        detect()