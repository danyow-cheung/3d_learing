import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVOrthographicCameras,look_at_view_transform,look_at_rotation,
    RasterizationSettings,MeshRenderer,MeshRasterizer,BlendParams,
    SoftSilhouetteShader,HardPhongShader,PointLights,TexturesVertex
)
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_mesh', default="./demo_results/0_mesh_sofa_0.887.obj")
parser.add_argument('--save_path', default='./demo_results/sofa_render.png')
parser.add_argument('--distance', default=1, help ='distance from camera to the object')
parser.add_argument('--elevation', default=150.0, help ='angle of elevation in degrees')
parser.add_argument('--azimuth', default=-10.0, help ='rotation of the camera')
args = parser.parse_args()

# 記載obj文件
verts,faces_idx,_ = load_obj(args.path_to_mesh)
faces = faces_idx.verts_idx

# 初始化每個體素
verts_rgb = torch.ones_like(verts)[None] 

textures = TexturesVertex(verts_features=verts_rgb.to(device= 'cuda:0'if torch.cuda.is_available() else 'cpu'))

sofa_mesh = Meshes(
    verts=[verts],
    faces=[faces],
    textures=textures
)

cameras = FoVOrthographicCameras()
blend_params = BlendParams(sigma=1e-4,gamma=1e-4)

raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1./1e-4 -1.)*blend_params.sigma,
    faces_per_pixel=100,
)
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings= raster_settings
    ),
    shader= SoftSilhouetteShader(blend_params=blend_params)
)

lights = PointLights(location=(2.0,2.0,-2.0))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader = HardPhongShader(cameras=cameras,lights=lights)
)

# 根據球面角度創建相機的位置
R,T = look_at_view_transform(distance,elevation,azimuth)

silhouette = silhouette_renderer(meshes_world = sofa_mesh,R=R,T=T)

image_ref = phong_renderer(meshes_world = sofa_mesh,R=R,T=T)
# 可視化
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(silhouette.squeeze()[..., 3])
plt.grid(False)
plt.subplot(1, 2, 2)
plt.imshow(image_ref.squeeze())
plt.grid(False)
plt.savefig(args.save_path)
