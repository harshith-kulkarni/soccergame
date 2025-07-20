# Soccer Game Detection Setup (Windows 11)

## 1. Install Anaconda
Download and install from: https://www.anaconda.com/download

## 2. Ensure Python is Installed
Recommended: Python 3.10 or 3.11  
Check version:
```sh
python --version
```

## 3. (Optional) Create a New Conda Environment
```sh
conda create -n soccergame python=3.11
conda activate soccergame
```

## 4. Install Required Packages
Check your CUDA version:
```sh
nvidia-smi
```
Then install PyTorch (replace cu121 with your CUDA version if needed):
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python deep-sort-realtime numpy
```

## 5. Verify PyTorch and CUDA
```sh
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```
https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view download best.pt from this to use model
## 6. Run the Script
```sh
python main.py
```
or
```sh
python second.py
```

---

**After running the verification, you should see output like:**
```
2.3.0+cu121
True
NVIDIA GeForce RTX 3050 6GB Laptop GPU
```