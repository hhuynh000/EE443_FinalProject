# visulization of tracking output
from argparse import ArgumentParser
import os
import numpy as np
import cv2 
from tqdm import tqdm

# get agrugments 
def get_args():
    parser = ArgumentParser(add_help=False, usage=usageMsg())
    parser.add_argument("data", nargs=2, help="Path to <images> <tracking>.")
    parser.add_argument('--help', action='help', help='Show this help message and exit')
    return parser.parse_args()

def usageMsg():
    return "python viz.py <images> <tracking>"

# assign different color for each id
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color 

# draw bounding box and corresponding label
def draw_box(img, tracking):
    label = tracking[1]
    x0 = tracking[3]
    x1 = tracking[3] + tracking[5]
    y0 = tracking[4]
    y1 = tracking[4] + tracking[6]
    
    start_point = (int(x0), int(y0))
    end_point = (int(x1), int(y1))
    color = get_color(label)
    img = cv2.putText(img, str(label), (int(x0), int(y0)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    return cv2.rectangle(img, start_point, end_point, color=color, thickness=2)

# generate video from img and tracking information
def gen_vid(image_path, track_path):
    video_name = 'tracking.mp4'
    if os.path.exists(video_name):
        print('Delete', video_name)
        os.remove(video_name)

    trackings = np.loadtxt(track_path, delimiter=',', dtype=None)
    num_images = len(os.listdir(image_path))
    height, width = 1080, 1920
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    for i in tqdm(range(num_images)):
        img_file = '{}.jpg'.format('%05d'%i)
        inds = trackings[:,2] == i
        cur_frame_tracking = trackings[inds]
        img = cv2.imread(os.path.join(image_path, img_file))
        for tracking in cur_frame_tracking:
            img = draw_box(img, tracking)
            
        video.write(img)

    video.release()
    cv2.destroyAllWindows() 
    

if __name__ == '__main__':
    args = get_args()
    if not args.data or len(args.data) < 2:
        print("Incorrect number of arguments. Must provide paths for the test (ground truth) and predicitons.")
    
    image_path = args.data[0]
    track_path = args.data[1]

    gen_vid(image_path, track_path)