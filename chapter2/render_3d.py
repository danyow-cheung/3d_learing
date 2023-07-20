import open3d
import os 
import sys 
import torch
import matplotlib.pyplot as plt 
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,PerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,

    )
from pytorch3d.renderer.mesh.shader import HardFlatShader
sys.path.append(os.path.abspath(''))
DATA_DIR = './data'
obj_filename = os.path.join(DATA_DIR,'cow_mesh/cow.obj')
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
mesh = load_objs_as_meshes([obj_filename],device=device)

'''
接下來我們將定義相機和光源。 
我們使用look_at_view_transform函數來映射易於理解的參數，
例如距相機的距離、仰角和方位角，以獲得旋轉（R）和平移（T）矩陣。 
R 和 T 變量定義了我們要放置相機的位置。 
lights 變量是一個點光源，其位置位於 [0.0, 0.0, -3.0] 處：
'''
R,T = look_at_view_transform(2.7,0,180)
cameras = PerspectiveCameras(device=device,R=R,T=T)
lights = PointLights(device=device,location=[[0.0,0.0,-3.0]])

raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardFlatShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)

images = renderer(mesh)
plt.figure(figsize=(10,10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.savefig('light_at_front.png')
plt.show()
'''
接下來，我們將光源的位置更改為網格的背面，看看會發生什麼。 渲染後的圖像如圖2.6所示。 在這種情況下，來自點光源的光不能與面向我們的任何網格面相交。 因此，我們在這裡可以觀察到的所有顏色都是由環境光造成的：
'''
lights,location = torch.tensor([0.0,0.0,+1.0],device=device)[None]
images = renderer(mesh,lights=lights)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.show()

'''
在下一個實驗中，我們將定義一個材料數據結構。 
在這裡，我們更改配置，使環境分量接近 0（實際上是 0.01）。 
由於點光源位於物體後方，並且環境光也被關閉，因此渲染的物體現在不反射任何光線。 
'''
materials = Materials(
    device=device,
    specular_color=[[0.0,1.0,0.0]],
    shininess=10.0,
    ambient_color=((0.01,0.01,0.01)),
)
images = renderer(mesh,lights=lights,materials=materials)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.show()
'''
在接下來的實驗中，我們將再次旋轉相機並重新定義光源位置，使光線能夠照射到牛的臉上。 
請注意，當我們定義材質時，我們將光澤度設置為 10.0。 
這個光澤度參數正是 Phong 光照模型中的 p 參數。 
Specular_color 為 [0.0, 1.0, 0.0]，這意味著表面主要在綠色分量中發亮。 
'''
R,T = look_at_view_transform(dist=2.7,elev=10,azim=150)
cameras = PerspectiveCameras(device=device,R=R,T=T)
lights.location = torch.tensor([[2.0,2.0,-2.0]],device=device)

materials = Materials(
    device=device,
    specular_color=[[0.0,1.0,0.0]],
    shininess=10.0
)
images = renderer(mesh, lights=lights,
materials=materials, cameras=cameras)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.show()
'''
在下一個實驗中，我們將把鏡面反射顏色更改為紅色並增加光澤度值。
'''
materials = Materials(
    device=device,
    specular_color=[[1.0,0.0,0.0]],
    shininess=20
)
images = renderer(mesh, lights=lights,
materials=materials, cameras=cameras)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.show()


aterials = Materials(
    device=device,
    specular_color=[[0.0, 0.0, 0.0]],
    shininess=0.0
)
images = renderer(mesh, lights=lights,
materials=materials, cameras=cameras)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.savefig('blue.png')
plt.axis("off")
plt.show()
# ----
# so眼訓