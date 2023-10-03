import os 
from glob import glob 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import argparse 

from serialization import load_model
from smplify.code.fit3d_utils import run_single_fit

MODEL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),'models')
MODEL_NEUTRAL_PATH = os.path.join(
    MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
)
print(MODEL_DIR)
print(MODEL_NEUTRAL_PATH)
# 讓我們設定最佳化方法所需的一些參數，並定義影像和結果所在的目錄。
viz = True 
n_betas = 10 
flength = 5000.0
pix_thsh = 25.0 
base_dir = os.path.dirname(__file__)+'/smplify'
img_dir = os.path.join(os.path.abspath(base_dir),'images/lsp')
data_dir = os.path.join(os.path.abspath(base_dir),'results/lsp')
print(img_dir)
print(data_dir)

outdir  = os.path.join(os.path.abspath(base_dir),'outputs')
if not os.path.exists(outdir):
    os.makedirs(outdir)

model = load_model(MODEL_NEUTRAL_PATH)
# 加载关节估计
est = np.load(os.path.join(data_dir,'est_joints.npz'))['est_joints']

img_paths = sorted(glob(os.path.join(img_dir, '*[0-9].jpg')))
for ind ,img_path in enumerate(img_paths):
    img = cv2.imread(img_path)
    joints = est[:2,:,ind].T
    conf = est[2,:,ind]

# 對於資料集中的每個影像，使用 run_single_fit 函數來擬合參數
# beta和theta。 在對類似於我們在上一節中討論的 SMPLify 目標函數的目標函數執行最佳化後，以下函數會傳回這些參數：
params,vis = run_single_fit(img,joints,conf,model)


if viz:
    plt.ion()
    plt.show()
    plt.subplot(121)
    plt.imshow(img[:,:,::-1])

