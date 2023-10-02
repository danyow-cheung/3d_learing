import cv2 
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.serialization import load_model

# 加載模型
m = load_model('../smplify/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

# 指派隨機的位置，形狀參數
m.pose[:] = np.random.rand(m.pose.size)*.2
m.betas[:] = np.random.rand(m.betas.size)*0.03 
m.pose[0] = np.pi 
# 我們現在創建一個渲染器並為其分配屬性，然後建立光源。
#  預設情況下，我們使用 OpenDR 渲染器，但您可以將其切換為使用 PyTorch3D 渲染器和光源。 
# 在此之前，請確保解決任何 Python 不相容問題。

rn = ColoredRenderer()
w,h = (640,480)
rn.camera = ProjectPoints(v=m,rt=np.zeros(3),
                          t=np.array([0,0,2.]),
                          f=np.array([w,w])/2.,
                          c=np.array([w,h])/2,
                          k=np.zeros(5))

rn.frustum = {'near':1.,'far':10.,'width':w,'height':h}
rn.set(v=m,f=m.f,bgcolor=np.zeros(3))
rn.vc = LambertianPointLight(
    f=m.f,
    v=rn.v,num_verts=len(m),
    light_pos = np.array([-1000,-1000,-2000]),
    vc = np.ones_like(m)*0.9,
    light_color = np.array([1.,1.,1.])
    )
cv2.imshow('renderer_SMPL',rn.r)
cv2.waitKey(0)
cv2.destroyAllWindows()
