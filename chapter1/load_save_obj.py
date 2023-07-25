import torch 
print(torch.__version__)
import open3d
from pytorch3d.io import load_obj

mesh_file = 'cube.obj'
mesh = open3d.io.read_triangle_mesh(mesh_file)
open3d.visulation.draw_geometries([mesh],mesh_show_wireframe = True,mesh_show_back_face =True)

vertices,faces,aux = load_obj(mesh_file)
print(type(vertices))
print(type(faces))
print(type(aux))

print(vertices)
print(faces)
print(aux)

