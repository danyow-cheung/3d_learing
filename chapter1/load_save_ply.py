'''
20230717 
每個 PLY 文件包含一個標頭部分和一個數據部分。 每個 ASCII PLY 文件的
第一行始終是 ply，這表明這是一個 PLY 文件。 
第二行format ascii 1.0表明該文件是Ascii類型並帶有版本號。 
任何以 comment 開頭的行都將被視為註釋行，
元素 vertex 8 行意味著 PLY 文件中的第一種數據是頂點，我們有八個頂點。
property float32 指的是數據類型

元素face 12行意味著這個PLY文件中的第二類數據是face類型，我們有12個面。 
屬性列表unit8 int32 vertex_indices顯示每個面將是頂點索引列表。
ply 文件的標頭部分始終以 end_header 行結束。
'''

# ply 文件數據部分的第二部分由 12 行組成，其中每一行都是一個面的記錄。
#  數字序列中的第一個數字表示該面具有的頂點數，後面的數字是頂點索引。 
# 頂點索引由 PLY 文件中聲明頂點的順序確定。

import open3d 
import torch 

from pytorch3d.io import load_ply 

mesh_file = 'cube.ply'
print('visualizing')
mesh = open3d.io.read_triangle_mesh(mesh_file)
open3d.visualization.draw_geometries([mesh],mesh_show_wireframe=True,mesh_show_back_face=True)
print('loading the same file with pytorch3d')

vertices,faces = load_ply(mesh_file)
print(type(vertices))
print(type(faces))
print(vertices)
print('--'*10)
print(faces)