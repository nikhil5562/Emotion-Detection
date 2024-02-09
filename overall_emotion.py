import argparse
import time
import os
from pathlib import Path
import numpy as np

import cv2
import csv
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from confidence_emotion import detect_emotion, init,emotions

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, save_one_box, create_folder
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

def detect(opt, csv_writer=None, folder_label=None):

    # Initialize the sum of probabilities for each emotion and frame count for each new video
    global emotion_sums, frame_count
    emotion_sums = {emotion: 0.0 for emotion in ("anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")}
    frame_count = 0
   
    source, view_img, imgsz, nosave, show_conf, save_path, show_fps = opt.source, not opt.hide_img, opt.img_size, opt.no_save, not opt.hide_conf, opt.output_path, opt.show_fps
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    # Directories
    create_folder(save_path)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    init(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load("weights/yolov7-tiny.pt", map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = ((0,52,255),(121,3,195),(176,34,118),(87,217,255),(69,199,79),(233,219,155),(203,139,77),(214,246,255))

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                images = []
                for *xyxy, conf, cls in reversed(det):
                    if show_conf:  # Add confidence to label
                        label = f'{names[int(cls)]} {conf:.2f}'
                    else:
                        label = f'{names[int(cls)]}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=opt.line_thickness)

                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]


                    images.append(im0.astype(np.uint8)[int(y1):int(y2), int(x1): int(x2)])
                

                if images:
                    frame_emotions = detect_emotion(images, show_conf)
                    frame_emotion_sums = {emotion: 0.0 for emotion in ("anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")}
    
                    for emotion_probabilities in frame_emotions:
                        for emotion, prob_str in emotion_probabilities:
                            prob = float(prob_str.strip('%')) / 100  # Convert percentage string to float
                            frame_emotion_sums[emotion] += prob  # Sum up emotions for the current frame
                            emotion_sums[emotion] += prob  # Sum up emotions for overall video

                    frame_count += 1
    
                    # Print out the emotions for the current frame
                    print(f"Frame {frame_count}:")
                    for emotion, sum_prob in frame_emotion_sums.items():
                        # Calculate the average if there are multiple faces detected
                        avg_prob = sum_prob / len(frame_emotions) if frame_emotions else 0
                        print(f"  {emotion}: {avg_prob:.2%}")

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_path != '' and not nosave:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if vid_cap else im0.shape[1]
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if vid_cap else im0.shape[0]
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    # Calculate and print overall emotion averages after processing all frames
    if frame_count > 0:
        emotion_averages = {emotion: sum_prob / frame_count for emotion, sum_prob in emotion_sums.items()}
        
        if emotion_averages:  # Ensure there are calculated averages to evaluate
            dominant_emotion = max(emotion_averages, key=emotion_averages.get)
            dominant_emotion_avg = emotion_averages[dominant_emotion]
            print(f"Dominant Emotion: {dominant_emotion.capitalize()} with an average probability of {dominant_emotion_avg:.2%}")
        else:
            print("No dominant emotion could be determined.")

        print("Overall emotions in the video:")
        for emotion, avg_prob in emotion_averages.items():
            print(f"{emotion.capitalize()}: {avg_prob:.2%}")


        # Write results to CSV
        # Inside your detect function, after calculating emotion_averages and dominant_emotion
        if csv_writer:
            video_name = Path(source).name
            # Ensure emotion_averages is calculated correctly up to this point
            csv_writer.writerow([folder_label, video_name] +
                                [f"{emotion_averages[emotion]*100:.2f}%" for emotion in emotions] +
                                [dominant_emotion])
    
    else:
        print("No frames were processed.")

    print(f"Done. ({time.time() - t0:.3f}s)")


def process_videos_in_folders(dataset_folder, output_folder, opt):
    # Define the path for the CSV file where results will be saved
    csv_file_path = os.path.join(output_folder, 'emotion_analysis_results.csv')

    # Open the CSV file to write the header and then process each video
    with open(csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # Define CSV header: label, video name, emotions, and dominant emotion
        csv_writer.writerow(['label', 'video_name'] + list(emotions) + ['dominant_emotion'])

        # List of subfolders corresponding to different emotion categories
        emotions_folders = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

        # Iterate through each emotion category subfolder
        for emotion_folder in emotions_folders:
            print(f"Currently processing folder: {emotion_folder}")  
            emotion_folder_path = os.path.join(dataset_folder, emotion_folder)

            # Check if the subfolder exists
            if not os.path.exists(emotion_folder_path):
                print(f"Emotion category '{emotion_folder}' not found in dataset folder.")
                continue

            # List all video files in the emotion category subfolder
            video_files = [f for f in os.listdir(emotion_folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

            # Iterate through each video file in the subfolder
            for video_file in video_files:
                video_path = os.path.join(emotion_folder_path, video_file)
                opt.source = video_path  # Set the source to the current video file
                # Define a unique output path for processed videos or frames
                video_output_folder = os.path.join(output_folder, emotion_folder)
                os.makedirs(video_output_folder, exist_ok=True) 
                opt.output_path = os.path.join(video_output_folder, Path(video_file).stem + "_processed" + Path(video_file).suffix)

                print(f"Processing video: {video_path}")

                # Call the detect function to process the video, passing the csv_writer to log results
                with torch.no_grad():
                    detect(opt=opt, csv_writer=csv_writer, folder_label=emotion_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='face confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--hide-img', action='store_true', help='hide results')
    parser.add_argument('--output-path', default="output.mp4", help='save location')
    parser.add_argument('--no-save', action='store_true', help='do not save images/videos')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--show-fps', default=False, action='store_true', help='print fps to console')
    opt = parser.parse_args()
    #check_requirements(exclude=('pycocotools', 'thop'))

    # Dataset folder containing subfolders for each emotion category
    dataset_folder = r"C:\Users\nikhi\Desktop\emotion_percentage\Emotion-Detection\test_dataset"
    # Output folder to save processed videos
    output_folder = r"C:\Users\nikhi\Desktop\emotion_percentage\Emotion-Detection\results"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        process_videos_in_folders(dataset_folder, output_folder, opt)