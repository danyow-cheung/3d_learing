import open3d
import torch 
import pytorch3d
from pytorch3d.io import load_obj
from scipy.spatial.transform import Rotation  as Rotation
from pytorch3d.renderer.cameras import PerspectiveCameras

# load meshs and visualized it with open3d 
mesh_file = 'cube.obj'
mesh = open3d.io.read_triangle_mesh(mesh_file)

open3d.visualization.draw_geometries([mesh],
                                     mesh_show_wireframe=True,
                                     mesh_show_back_face=True)

'''
define a camera variable as a pytorch3d perspectiveCamera 针孔相机模型
'''
# define a mini-batch of 8 camera 
image_size = torch.ones(8,2)
image_size[:,0] = image_size[:,0]*1024 
image_size[:,1] = image_size[:,1]*512

focal_length = torch.ones(8,2)
focal_length[:,0] = focal_length[:,0]*1200
focal_length[:,1] = focal_length[:,1]*300

principal_point = torch.ones(8,2)
principal_point[:,0] = principal_point[:,0]*512 
principal_point[:,1] = principal_point[:,1]*512 

R = Rotation.from_euler('zyx',[[n*5,n,n] for n in range(-4,4,1)],
                        degrees=True).as_matrix()

R = torch.from_numpy(R)
T = [[n,0,0] for n in range(-4,4,1)]
T = torch.FloatTensor(T)

camera = PerspectiveCameras(focal_length=focal_length,
                            principal_point=principal_point,
                            in_ndc=False,
                            image_size=image_size,
                            R = R,
                            T = T,
)
'''
一旦我們定義了camera變量，我們就可以調用get_world_to_view_transform類成員方法來獲取Transform3d對象world_to_view_transform。 
然後我們可以使用transform_points成員方法從世界坐標轉換為相機視圖坐標。 
同樣，我們也可以使用get_full_projection_transform成員方法獲取一個Transform3d對象，
該對像用於世界坐標到屏幕坐標的轉換：
'''
world_to_view_transform = camera.get_world_to_view_transform()
world_to_screen_transform = camera.get_full_projection_transform()

#load meshs using pytorch3d 
vertices,faces ,aux = load_obj(mesh_file)

world_to_view_vertices = world_to_view_transform.transform_points(vertices)
world_to_screen_vertices = world_to_screen_transform.transform_points(vertices)

print('world_to_view_vertices = ', world_to_view_vertices)
print('world_to_screen_vertices = ', world_to_screen_vertices)


