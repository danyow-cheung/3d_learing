import open3d
import os 
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures.meshes import join_meshes_as_batch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import numpy as np 

if torch.cuda.is_available():
    device =torch.device('cuda:0')
else:
    device = torch.device('cpu')

mesh_names = ['cube.obj', 'diamond.obj', 'dodecahedron.obj']
data_path = './data'
for mesh_name in mesh_names:
    mesh = open3d.io.read_triangle_mesh(os.path.join(data_path,mesh_name))
    open3d.visualization.draw_geometries([mesh],
                                         mesh_show_wireframe =True, 
                                         mesh_show_back_face =True,)
    
'''
使用 PyTorch3D 加載相同的網格並構建網格列表，即 mesh_list 變量：
'''
mesh_list = list()
for mesh_name in mesh_names:
    mesh = load_objs_as_meshes([os.path.join(data_path,mesh_name)],device=device)
    mesh_list.append(mesh)
mesh_batch = join_meshes_as_batch(mesh_list, include_extures = False)

# how to return vertices and faces in a list format from a mini-batch:
vertex_list = mesh_batch.verts_list()
face_list = mesh_batch.faces_list()
# return vertices and faces in the padded format
vertex_padded = mesh_batch.verts_padded()
# get vertices and faces in the packed format 
vertex_packed = mesh_batch.verts_packed()
num_vertices = vertex_packed.shape[0]

'''
我們將 mesh_batch 變量視為三個對象的真實網格模型。 然後，我們將模擬三個網格的噪聲和位移版本。 第一步，我們要克隆地面實況網格模型：
'''
mesh_batch_noisy = mesh_batch.clone()
# 然後我們定義一個motion_gt變量來表示相機位置和原點之間的位移
motion_gt = np.array([3,4,5])
motion_gt =torch.as_tensor(motion_gt)

motion_gt = motion_gt[None,:]
motion_gt = motion_gt.to(device)
# 為了模擬嘈雜的深度相機觀測結果，我們生成一些隨機高斯噪聲，其平均值等於motion_gt。 
# 使用 offset_verts PyTorch3D 函數將噪聲添加到 mesh_batch_noisy 中：
noise = (0.1**0.5)*torch.randn(mesh_batch_noisy.verts_packed().shape).to(device)
noise = noise+motion_gt
mesh_batch_noisy = mesh_batch_noisy.offset_verts(noise).detach()
'''
為了估計相機和原點之間的未知位移，我們將製定一個優化問題。
 首先，我們將定義motion_estimate優化變量。 torch.zeros 函數將創建一個全零的 PyTorch 張量。 
 請注意，我們將 require_grad 設置為 true。 這意味著當我們從損失函數運行梯度反向傳播時，
 我們希望 PyTorch 自動計算該變量的梯度：
'''
motion_estimate = torch.zeros(motion_gt.shape,device=device,requires_grad=True)
# 接下來，我們將定義一個學習率為 0.1 的 PyTorch 優化器。 
# 通過將變量列表傳遞給優化器，我們指定此優化問題的優化變量。 這裡，優化變量是motion_estimate變量：
optimizer = torch.optim.SGD([motion_estimate],lr=0.1,momentum=0.9)
# 在計算損失函數的過程中，我們從兩個網格中隨機採樣 5,000 個點併計算它們的 Chamfer 距離。 倒角距離是兩組點之間的距離。 我們將在後面的章節中更詳細地討論這個距離函數：
for i in range(0,200):
    optimizer.zero_grad()
    current_mesh_batch = mesh_batch.offset_verts(motion_estimate.repeat(num_vertices,1))
    sample_trg = sample_points_from_meshes(current_mesh_batch,5000)
    sample_src = sample_points_from_meshes(mesh_batch_noisy,5000)

    loss,_ = chamfer_distance(sample_trg,sample_src)
    loss.backward()
    optimizer.step()
    print('i = ', i, ', motion_estimation = ', motion_estimate)

    