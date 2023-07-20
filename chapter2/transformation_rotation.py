import torch 
from pytorch3d.transforms.so3 import so3_exp_map,so3_log_map,hat_inv,hat 
if torch.cuda.is_available():
    device =torch.device('cuda:0')
else:
    device = torch.device('cpu')

# 接下來，我們將定義四輪旋轉的小批量。 
# 這裡，每次旋轉都由一個 3D 向量表示。 矢量的方向代表旋轉軸，矢量的幅度代表旋轉角度：
log_rot = torch.zeros([4,3],device=device)
log_rot[0,0] = 0.001
log_rot[0,1] = 0.0001
log_rot[0,2] = 0.0002

log_rot[1,0]=0.0001
log_rot[1,1] = 0.001
log_rot[1,2] = 0.0002

log_rot[2,0] - 0.0001
log_rot[2,1] = 0.0002
log_rot[2,2] = 0.001

log_rot[3,0] = 0.001
log_rot[3,1] = 0.002
log_rot[3,2] = 0.003
# log_rot 的形狀為 [4, 3]，其中 4 是批量大小，每次旋轉由 3D 向量表示。 我們可以使用 PyTorch3D 中的帽子運算符將它們轉換為 3 x 3 斜對稱矩陣表示，如下所示：

log_rot_hat = hat(log_rot)
print(log_rot_hat.shape)
print(log_rot_hat)

log_rot_copy = hat_inv(log_rot_hat)
print(log_rot_copy.shape)
rotation_matrices = so3_exp_map(log_rot)
print(rotation_matrices)

log_rot_again = so3_log_map(rotation_matrices)
print(log_rot_again)
