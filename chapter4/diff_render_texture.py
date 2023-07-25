import os 
import torch
import numpy as np 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from skimage import img_as_ubyte 
from pytorch3d.io import load_objs_as_meshes

from pytorch3d.renderer import (
    FoVPerspectiveCameras,look_at_view_transform,look_at_rotation,
    RasterizationSettings,MeshRenderer,MeshRasterizer,
    BlendParams,SoftSilhouetteShader,HardPhongShader,PointLights,
    SoftPhongShader
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

output_dir = './result_cow'
# load the model of a toy from cow.obj file 
obj_filename = './data/cow_mesh/cow.obj'
cow_mesh = load_objs_as_meshes([obj_filename],device=device)
cameras = FoVPerspectiveCameras(device=device)
lights = PointLights(device=device,location=((2.0,2.0,-2.0),))

blend_params = BlendParams(sigma=1e-4,gamma=1e-4)
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1./1e-4-1.)*blend_params.sigma,
    faces_per_pixel=100,
)
renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(blend_params=blend_params)
)

sigma  = 1e-4 
raster_setting_soft = RasterizationSettings(image_size=256,blur_radius=np.log(1./1e-4-1.)*sigma,faces_per_pixel=50)


renderer_textured = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_setting_soft
    ),
    shader= SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)


raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1,
)
phong_renderer = MeshRasterizer(
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader = HardPhongShader(device=device,cameras=cameras,lights=lights)
)

'''
接下來，我們將定義一個相機位置以及相應的相機旋轉和位置。
這將是拍攝觀察到的圖像的相機位置。 與前面的示例一樣，我們優化相機方向和位置，而不是對象方向和位置。
此外，我們假設相機始終指向物體。 因此，我們只需要優化相機位置即可：
'''
distance = 3 
elevation = 50.0 
azimuth = 0.0
R, T = look_at_view_transform(distance, elevation,
azimuth, device=device)
'''
創造觀察圖片然後保存
'''
silhouette = renderer_silhouette(meshes_world=cow_mesh,R=R,T=T)
image_ref = phong_renderer(meshes_world=cow_mesh,R=R,T=T)
silhouette = silhouette.cpu().numpy()
image_ref = image_ref.cpu().numpy()
plt.figure(figsize=(10, 10))
plt.imshow(silhouette.squeeze()[..., 3])
plt.grid(False)

plt.close()
plt.figure(figsize=(10, 10))
plt.imshow(image_ref.squeeze())
plt.grid(False)
plt.close()
'''
我們修改 Model 類的定義如下。 與上一個示例相比最顯著的變化是，現在我們將渲染 alpha 通道圖像和 RGB 圖像，
並將它們與觀察到的圖像進行比較。 對 alpha 通道和 RGB 通道的均方損失進行加權，得出最終損失值
'''
class Model(nn.Module):
    def __init__(self,meshes,renderer_silhouette,renderer_textured,image_ref,weight_silhouette,weight_texture):
        super().__init__()
        self.meshes = meshes 
        self.device = meshes.device
        self.renderer_silhouette = renderer_silhouette
        self.renderer_textured = renderer_textured
        self.weight_silhouette = weight_silhouette
        self.weight_texture = weight_texture

        image_ref_silhouette = torch.from_numpy((image_ref[...,:3].max(-1)!=1).astpye(np.float32))
        self.register_buffer('image_ref_silhouette',image_ref_silhouette)

        image_ref_textured = torch.from_numpy((image_ref[...,:3]).astype(np.float32))
        self.register_buffer('image_ref_textured', image_ref_textured)

        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([3.0,6.9,+2.5],dtype=np.float32).to(meshes.device)))
        
    def forward(self):
        # render the image using the updated camera postion Based on the new position of the camera we 
        # calculate the rotation and translation matrices 
        R = look_at_rotation(self.camera_position[None,:],device=self.device)
        