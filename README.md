# eye-contact-cnn (macOS CPU fork)

This repository provides a deep convolutional neural network model trained to detect moments of eye contact in egocentric view. The original model was trained on over 4 million facial images of > 100 young individuals during natural social interactions, achieving accuracy comparable to trained clinical human annotators. This fork adapts everything to run on macOS (M1/M2) CPU (or MPS if available) under Python 3.12+.

![Teaser](teaser.gif)

---

## 🔧 Environment & Requirements

### 1. Clone & venv (recommended)

```bash
git clone https://github.com/<your-username>/eye-contact-cnn.git
cd eye-contact-cnn

# create a clean Python venv
python3 -m venv venv
source venv/bin/activate

# upgrade pip and install everything
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
> **Important (macOS users)**:
On Apple Silicon (M1/M2) you still need a prebuilt dlib.
Either comment out dlib in requirements.txt and install manually: 
```bash
conda install -c conda-forge dlib=19.24.0
```
> Or install it into a separate conda env before activating your venv.

---

### 2. (Alternative) Conda-only setup

```bash
conda create -n eyecontact python=3.12 -y
conda activate eyecontact

# install binary deps
conda install -c conda-forge \
    dlib=19.24.0 \
    numpy=1.26.2 pandas=1.5.3 pillow=9.5.0 \
    opencv=4.7.0

# install the rest from PyPI
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### ▶️ Usage
By default every frame is processed, progress is printed live, and output video are saved into an output_videos/ folder (created automatically).

```bash
# Run and pick source (webcam or file) via on-screen prompt
python demo.py

# To disable the live display window and speed processing:
python demo.py -display_off

# If you have a pre-computed face CSV:
python demo.py --face path/to/face_detections.txt -save_text
```

## Notes
- Output eye contact score ranges [0, 1] and score above 0.9 is considered confident.
- To further improve the result, smoothing the output is encouraged as it can help removing outliers caused by eye blinks, motion blur etc.

## 📖 Citation

This repository is a **macOS CPU fork** of the original [eye-contact-cnn](https://github.com/rehg-lab/eye-contact-cnn). 

If you use the core model or data from the original work, please cite:

```bibtex
@article{chong2020,
  title={Detection of eye contact with deep neural networks is as accurate as human experts},
  url={osf.io/5a6m7},
  DOI={10.31219/osf.io/5a6m7},
  author={Chong, Eunji and Clark-Whitney, Elysha and Southerland, Audrey and Stubbs, Elizabeth
          and Miller, Chanel and Ajodan, Eliana L and Silverman, Melanie R and Lord, Catherine
          and Rozga, Agata and Jones, Rebecca M and et al.},
  year={2020},
  publisher={OSF Preprints}
}
```
Link to the paper:
[here](https://nature-research-under-consideration.nature.com/users/37265-nature-communications/posts/60730-detection-of-eye-contact-with-deep-neural-networks-is-as-accurate-as-human-experts)

## 👩🏻‍💻👨🏽‍💻 Key Improvements in This Fork
- Python 3.12 compatibility
- M1/M2 Apple Silicon compatibility (CPU and MPS)
- Automatic output saving to output_videos/
- Progress printed live
- Cleaner setup with or without dlib

---

This version is fully streamlined for macOS development and testing.
