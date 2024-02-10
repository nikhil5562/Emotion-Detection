import argparse
import time
import os
from pathlib import Path
import numpy as np
import cv2
import csv
import torch
import torch.backends.cudnn as cudnn

from confidence_emotion import detect_emotion, init, emotions
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, save_one_box, create_folder
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


def check_imshow():
    return True

# Define colors for each emotion
emotion_colors = {
    "anger": (0, 0, 255),
    "contempt": (255, 0, 0),
    "disgust": (0, 255, 0),
    "fear": (255, 255, 0),
    "happy": (0, 255, 255),
    "neutral": (255, 0, 255),
    "sad": (192, 192, 192),
    "surprise": (128, 0, 128)
}

def detect(opt, csv_writer=None, folder_label=None):
    global emotion_sums, frame_count, total_faces_detected
    emotion_sums = {emotion: 0.0 for emotion in emotions}
    frame_count = 0
    total_faces_detected = 0

    source, view_img, imgsz, nosave, show_conf, save_path, show_fps = opt.source, not opt.hide_img, opt.img_size, opt.no_save, not opt.hide_conf, opt.output_path, opt.show_fps
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    create_folder(save_path)
    set_logging()
    device = select_device(opt.device)
    init(device)
    half = device.type != 'cpu'

    model = attempt_load("weights/yolov7-tiny.pt", map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                images = []
                emotion_probabilities_frame = []
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    images.append(im0.astype(np.uint8)[int(y1):int(y2), int(x1): int(x2)])

                if images:
                    frame_emotions = detect_emotion(images, show_conf)
                    total_faces_detected += len(images)
                    for emotion_probabilities in frame_emotions:
                        for emotion, prob_str in emotion_probabilities:
                            prob = float(prob_str.strip('%')) / 100
                            emotion_probabilities_frame.append((emotion, prob))
                            emotion_sums[emotion] += prob

                    dominant_emotion, dominant_prob = max(emotion_probabilities_frame, key=lambda item: item[1])
                    color = emotion_colors[dominant_emotion]

                    for *xyxy, _, _ in reversed(det):
                        label = f'{dominant_emotion} {dominant_prob:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=color, line_thickness=5)

                    # Increase frame count
                    frame_count += 1
                 # Display the image
                if view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
                        break

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

    if frame_count > 0:
        # Calculate and print overall emotion averages after processing all frames
        emotion_averages = {emotion: sum_prob / total_faces_detected for emotion, sum_prob in emotion_sums.items()}
        
        if emotion_averages:  # Ensure there are calculated averages to evaluate
            dominant_emotion = max(emotion_averages, key=emotion_averages.get)
            dominant_emotion_avg = emotion_averages[dominant_emotion]
            print(f"Dominant Emotion: {dominant_emotion.capitalize()} with an average probability of {dominant_emotion_avg:.2%}")
            
            print("Overall emotions in the video:")
            for emotion, avg_prob in emotion_averages.items():
                print(f"{emotion.capitalize()}: {avg_prob:.2%}")

        # Write results to CSV
        if csv_writer:
            video_name = Path(source).name
            csv_writer.writerow([folder_label, video_name] +
                                [f"{emotion_averages[emotion]*100:.2f}%" for emotion in emotions] +
                                [dominant_emotion])

    else:
        print("No frames were processed.")

    # Release video writer and close windows
    if vid_writer:
        vid_writer.release()
    cv2.destroyAllWindows()

    print(f"Done. ({time.time() - t0:.3f}s)")


def process_videos_in_folders(dataset_folder, output_folder, opt):
    csv_file_path = os.path.join(output_folder, 'emotion_analysis_results.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['label', 'video_name'] + list(emotions) + ['dominant_emotion'])

        emotions_folders = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        for emotion_folder in emotions_folders:
            print(f"Currently processing folder: {emotion_folder}")
            emotion_folder_path = os.path.join(dataset_folder, emotion_folder)

            if not os.path.exists(emotion_folder_path):
                print(f"Emotion category '{emotion_folder}' not found in dataset folder.")
                continue

            video_files = [f for f in os.listdir(emotion_folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
            for video_file in video_files:
                video_path = os.path.join(emotion_folder_path, video_file)
                opt.source = video_path
                video_output_folder = os.path.join(output_folder, emotion_folder)
                os.makedirs(video_output_folder, exist_ok=True)
                opt.output_path = os.path.join(video_output_folder, Path(video_file).stem + "_processed" + Path(video_file).suffix)

                print(f"Processing video: {video_path}")
                with torch.no_grad():
                    detect(opt=opt, csv_writer=csv_writer, folder_label=emotion_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--hide-img', action='store_true', help='hide results')
    parser.add_argument('--output-path', default="output.mp4", help='output path')
    parser.add_argument('--no-save', action='store_true', help='do not save images/videos')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    parser.add_argument('--show-fps', action='store_true', help='show FPS')
    opt = parser.parse_args()

    dataset_folder = r"C:\Users\nikhi\Desktop\Emotion-Detection\test_dataset"
    output_folder = r"C:\Users\nikhi\Desktop\Emotion-Detection\results"
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        process_videos_in_folders(dataset_folder, output_folder, opt)     