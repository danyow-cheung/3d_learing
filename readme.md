# 3d_learning
> 要安装gpu好像看到自己失败的未来
> 无所谓，现在的我已经会少一块来减枝
## consist of 3 parts 

**Parts 1:**
1. 3d data processing
2. 3d computer vision and geometry幾何學

**Parts 2:**
1. Fitting Deformable Mesh Models to Raw Point Clouds
2. Learning Object Pose Detection and Tracking by Differentiable Rendering 
3. Chapter 5, Understanding Differentiable Volumetric Rendering
4. Exploring Neural Radiance Fields (NeRF)

**Parts 3:**
1. Exploring Controllable Neural Feature Fields
2. Modeling the Human Body in 3D
3. Performing End-to-End View Synthesis with SynSin
4. Mesh R-CNN
   

在colab敲代碼需要的pytorch3d安裝方法
```
import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
!pip install fvcore iopath
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html
```

# 總結
有意義的開源項目
- [Mesh-RCNN](https://github.com/facebookresearch/meshrcnn.git)
- [giraffe](https://github.com/autonomousvision/giraffe)
- [synsin](https://github.com/facebookresearch/synsin.git)
- 