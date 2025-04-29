import argparse
import os
import random
import time
import tkinter as tk
from tkinter import filedialog

import dlib
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import model_static
from PIL import Image, ImageDraw, ImageFont
from colour import Color

# Pick device: MPS on Apple Silicon, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# Video‐source helper
# -----------------------------------------------------------------------------
class VideoSource:
    def __init__(self):
        src, self.is_webcam = self._select_source()
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self._last_time = time.perf_counter()

    def _select_source(self):
        print("Select video source:")
        print("  1) Use Webcam")
        print("  2) Choose video file")
        choice = input("Enter 1 or 2: ").strip()
        if choice == '2':
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Select a video file",
                filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")]
            )
            if path:
                print(f"Selected video: {path}")
                return path, False
            else:
                print("No file chosen; defaulting to webcam.")
                return 0, True
        else:
            return 0, True

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # mirror webcam for a more natural selfie view
        if self.is_webcam:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        self.cap.release()

# -----------------------------------------------------------------------------
# Argument parsing (no more --video)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--face',         type=str,
                    help='path to face‐detection CSV (skip for DLIB)')
parser.add_argument('--model_weight', type=str,
                    default='data/model_weights.pkl',
                    help='path to model weights file')
parser.add_argument('--jitter',       type=int, default=0,
                    help='number of bbox jitters to average')
parser.add_argument('-save_vis',      action='store_true', default=True,
                    help='save output video (enabled by default)')
parser.add_argument('-save_text',     action='store_true',
                    help='save scores to text')
parser.add_argument('-display_off',   action='store_true',
                    help='suppress real‐time display')
args = parser.parse_args()

CNN_FACE_MODEL = 'data/mmod_human_face_detector.dat'

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def bbox_jitter(l, t, r, b):
    cx, cy = (l + r) / 2.0, (t + b) / 2.0
    scale = random.uniform(0.8, 1.2)
    w, h = (r - l) * scale, (b - t) * scale
    return cx - w/2, cy - h/2, cx + w/2, cy + h/2

def drawrect(dc, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    dc.line(pts, fill=outline, width=width)

# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------
def run(face_csv, model_weight, jitter, vis, display_off, save_text):
    # 1) Select video source
    vs = VideoSource()
    cap = vs.cap
    video_path = vs.src if not vs.is_webcam else None
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)
    if video_path:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Total frames to process: {total_frames}")
    else:
        total_frames = None

    # 2) Prepare optional outputs
    if save_text:
        if video_path:
            base_name = os.path.basename(video_path)
            name_without_ext = os.path.splitext(base_name)[0]
            txt_name = f"{name_without_ext}_scores.txt"
        else:
            txt_name = "webcam_scores.txt"
        txt_path = os.path.join(output_dir, txt_name)              
        f = open(txt_path, 'w')                                    

    # Set up video writer - always save video output
    vis = True  # Force video saving
    if video_path:
        base_name = os.path.basename(video_path)
        name_without_ext = os.path.splitext(base_name)[0]
        out_name = f"{name_without_ext}_eye_contact.mp4"
    else:
        out_name = "webcam_eye_contact.mp4"
    
    out_path = os.path.join(output_dir, out_name)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    
    # Try different codecs based on what might be available
    try:
        # First try MP4V codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outvid = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        # Test if the video writer was initialized properly
        if not outvid.isOpened():
            raise Exception("Failed to open with mp4v codec")
            
    except Exception as e:
        print(f"Could not use mp4v codec: {e}")
        try:
            # Fall back to XVID codec
            out_path = os.path.join(output_dir, out_name.replace('.mp4', '.avi'))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            outvid = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            
            if not outvid.isOpened():
                raise Exception("Failed to open with XVID codec")
                
        except Exception as e2:
            print(f"Could not use XVID codec: {e2}")
            # Last resort - MJPG codec
            out_path = os.path.join(output_dir, out_name.replace('.mp4', '.avi'))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            outvid = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    print(f"Output video will be saved to: {out_path}")

    # 3) Face‐detection setup
    if face_csv:
        mode = 'GIVEN'
        cols = ['frame','left','top','right','bottom']
        df = pd.read_csv(face_csv, names=cols, index_col=0)
        df['left']  -= (df['right']-df['left'])*0.2
        df['right'] += (df['right']-df['left'])*0.2
        df['top']   -= (df['bottom']-df['top'])*0.1
        df['bottom']+= (df['bottom']-df['top'])*0.1
        df = df.astype(int)
    else:
        mode = 'DLIB'
        cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)

    # 4) Load model
    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])

    model = model_static(model_weight)
    model.load_state_dict(
        torch.load(model_weight, map_location="cpu"),
        strict=False
    )
    model.to(device)
    model.eval()

    # 5) Visualization settings
    red    = Color("red")
    colors = list(red.range_to(Color("green"), 10))
    font   = ImageFont.truetype("data/arial.ttf", 40)

    # 6) Frame loop
    frame_cnt = 0
    while True:
        frame = vs.read()
        if frame is None:
            break
        frame_cnt += 1
        if total_frames:
            pct = frame_cnt / total_frames * 100
            print(f"Progress: {frame_cnt}/{total_frames} ({pct:6.2f}%)", end='\r')
        else:
            print(f"Processed frame {frame_cnt}", end='\r')

        # RGB for PIL / DLIB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = []

        if mode == 'DLIB':
            dets = cnn_face_detector(rgb, 1)
            for d in dets:
                l, r = d.rect.left(),  d.rect.right()
                t, b = d.rect.top(),   d.rect.bottom()
                w, h = r-l, b-t
                l, r = l - .2*w, r + .2*w
                t, b = t - .1*h, b + .1*h
                bboxes.append([l,t,r,b])
        else:
            if frame_cnt in df.index:
                row = df.loc[frame_cnt]
                bboxes.append([row.left, row.top, row.right, row.bottom])

        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)

        for b in bboxes:
            crop = pil.crop(b)
            x = test_transforms(crop).unsqueeze(0).to(device)
            if jitter > 0:
                extras = []
                for _ in range(jitter):
                    bj = bbox_jitter(*b)
                    extras.append(test_transforms(pil.crop(bj)).unsqueeze(0))
                x = torch.cat([x] + extras, dim=0).to(device)

            out = model(x)
            if jitter > 0:
                out = out.mean(0, keepdim=True)
            score = torch.sigmoid(out).item()

            idx = min(int(round(score*9)), 9)
            drawrect(draw, [(b[0], b[1]), (b[2], b[3])],
                     outline=colors[idx].hex, width=5)
            draw.text((b[0], b[3]), f"{score:.2f}",
                      fill=(255,255,255,128), font=font)
            if save_text:
                f.write(f"{frame_cnt},{score:.4f}\n")

        # convert back for display/save
        out_frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        
        # Always save the frame (vis is always True now)
        outvid.write(out_frame)
        
        # Display unless disabled
        if not display_off:
            cv2.imshow("Eye Contact Analysis", out_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    # 7) Cleanup
    outvid.release()
    if save_text: 
        f.close()
    vs.release()
    cv2.destroyAllWindows()
    print("\nDONE! Video saved to:", out_path)

if __name__ == "__main__":
    run(args.face, args.model_weight,
        args.jitter, args.save_vis,
        args.display_off, args.save_text)