import os 
import sys 
import torch
from pytorch3d.io import load_ply,save_ply
from pytorch3d.io import load_obj,save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (chamfer_distance,mesh_edge_loss,mesh_laplacian_smoothing,mesh_normal_consistency)
import numpy as np 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow")

verts,faces = load_ply('pedestrain.ply')
verts = verts.to(device)
faces = faces.to(device)
# 正則化
center = verts.mean(0)
verts = verts - center 
scale = max(verts.abs().max(0)[0])
verts = verts / scale
verts = verts[None,:,:]
src_mesh = ico_sphere(4,device) 
'''
在下一步中，我們要定義變形_verts 變量。 
deform_verts 是頂點位移張量，其中對於 src_mesh 中的每個頂點，都有一個三維向量的頂點位移。 
我們將優化 deform_verts 以找到最佳的可變形網格：
'''
src_vert = src_mesh.verts_list()

deform_verts = torch.full(src_vert[0].shape,0.0,device=device,requires_grad=True)

optimizer = torch.optim.SGD([deform_verts],lr=1.0,momentum=0.9)
'''
我們為不同的損失函數定義一批權重。 正如我們所提到的，我們需要多個損失函數，
包括主損失函數和正則化損失函數。 最終的損失將是不同損失函數的加權和。 這是我們定義權重的地方：
'''
w_chamfer = 1.0 
w_edge = 1.0 
w_normal = 0.01
w_laplacian = 0.1 

# 開始訓練
for i in range(0,2000):
    print('i=',i)
    optimizer.zero_grad()
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    sample_trg = verts 
    sample_src = sample_points_from_meshes(new_src_mesh,verts.shape[1])
    loss_chamfer ,_ = chamfer_distance(sample_trg,sample_src)

    loss_edge = mesh_edge_loss(new_src_mesh)
    loss_normal = mesh_normal_consistency(new_src_mesh)
    loss_laplacian = mesh_normal_consistency(new_src_mesh,method='uniform')
    loss =(
        loss_chamfer *w_chamfer
        + loss_edge*w_edge
        + loss_normal *w_normal
        + loss_laplacian*w_laplacian
    )
    loss.backward()
    optimizer.step()

final_verts , final_faces = new_src_mesh.get_mesh_verts_faces(0)
final_verts = final_verts *scale + center
save_ply("deform1.ply", final_verts, final_faces, ascii=True)