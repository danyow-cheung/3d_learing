import os 
import torch 
import numpy as np 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from skimage import img_as_ubyte 
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVOrthographicCameras,look_at_view_transform,look_at_rotation,RasterizationSettings,MeshRenderer,MeshRasterizer,BlendParams,SoftSilhouetteShader,HardPhongShader,PointLights,TexturesVertex

output_dir = './result_teapot'

verts,faces_idx ,_ = load_obj('./data/teapot.obj')
faces = faces_idx.verts_idx

verts_rgb = torch.ones_like(verts)[None] 
textures = TexturesVertex(verts_features=verts_rgb)

teapot_mesh = Meshes(
    verts= [verts],
    faces = [faces],
    textures=textures
)

cameras = FoVOrthographicCameras()

blend_params = BlendParams(sigma=1e-4,gamma=1e-4)
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1./1e-4-1.)*blend_params.sigmas,
    faces_per_pixel=100,

)

silhouette_renderer = MeshRenderer(
    rasterizer=MeshRenderer(
        cameras = cameras,
        raster_settings = raster_settings,
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)

)

raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1,
)
lights = PointLights(location=(2.0,2.0,-2.0))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings= raster_settings
    ),
    shader=HardPhongShader(cameras=cameras,lights=lights)
)

distance = 3 
elevation = 50.0 
azimuth = 0.0 
R,T = look_at_view_transform(distance,elevation,azimuth)
'''
現在，我們從該相機位置生成圖像 image_ref。 
image_ref 函數有四個通道，RGBA – R 表示紅色，G 表示綠色，B 表示藍色，A 表示 alpha 值。 
image_ref函數也保存為target_rgb.png以供我們後期檢查：
'''
silhouette = silhouette_renderer(meshes_world = teapot_mesh,R=R,T=T)
image_ref = phong_renderer(meshes_world = teapot_mesh,R=R,T=T)
silhouette = silhouette.cpu().numpy()
image_ref = image_ref.cpu().numpy()
plt.figure(figsize=(10, 10))
plt.imshow(silhouette.squeeze()[..., 3])  # only plot the alpha channel of the RGBA image
plt.grid(False)
plt.savefig(os.path.join(output_dir, 'target_silhouette.png'))
plt.close()
plt.figure(figsize=(10, 10))
plt.imshow(image_ref.squeeze())
plt.grid(False)
plt.savefig(os.path.join(output_dir, 'target_rgb.png'))
plt.close()

'''
In the next step, we are going to define a Model class. This Model class is derived from torch. nn.Module; thus, as with many other PyTorch models, automatic gradient computations can be enabled for Model.
'''
class Model(nn.Module):
    def __init__(self, meshes,renderer,image_ref) -> None:
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
    
        image_ref = torch.from_numpy((image_ref[...,:3].max(-1)!=1).astype(np.float32))

        self.register_buffer('image_ref',image_ref)
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([3.0,6.9,+2.5],dtype=np.float32)).to(meshes.device)
        )

    def forward(self):
        R = look_at_rotation(self.camera_position[None,:],device=self.device)
        