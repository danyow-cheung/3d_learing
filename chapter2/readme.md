# 3D computer vision and geometry 
在本章中，我們將學習 3D 計算機視覺和幾何的一些基本概念，這些概念對於本書後面的章節特別有用。 我們將首先討論什麼是渲染、光柵化和著色。 我們將介紹不同的照明模型和著色模型，例如點光源、定向光源、環境照明、漫射、高光和光澤。 我們將通過一個編碼示例來使用不同的光照模型和參數渲染網格模型。

## Exploring the basic concepts of rendering ,rasterization and shading
**渲染**是一个将相机周围世界的 3D 数据模型作为输入和输出图像的过程。 它是现实世界中相机形成图像的物理过程的近似值。 通常，3D 数据模型是网格。 在这种情况下，渲染通常使用光线追踪来完成：

因此，渲染过程通常可以分为两个阶段——<u>光栅化</u>和<u>着色</u>。 光线追踪过程是一个典型的光栅化过程——即为每个图像像素寻找相关几何对象的过程。 着色是获取光栅化输出并计算每个图像像素的像素值的过程。

有個接口`pytorch3d.renderer.mesh.rasterize_meshes.rasterize_meshes`通常为每个图像像素计算以下四件事：
- pix_to_face 是射线可能相交的面索引列表。
- zbuf 是这些面的深度值的列表。
- bary_coords 是每个面和射线的交点的重心坐标列表。
- pix_dists 是像素（x 和 y）与光线相交的所有面上的最近点之间的带符号距离的列表。 该列表的值可以取负值，因为它包含有符号距离。

### barycentric coordinates重心坐标
对于与网格面共面的每个点，该点的坐标始终可以写为,网格面三个顶点坐标的线性组合

### Light source models 
現實世界中的光傳播可能是一個複雜的過程。 著色中通常使用光源的幾種近似值以減少計算成本：
- ambient lighting 
    第一個假設是環境照明，我們假設經過充分反射後存在一些背景光輻射，使得它們通常來自各個方向，在所有圖像像素處具有幾乎相同的幅度。

- point light sources 
    我們通常使用的另一個假設是某些光源可以被認為是點光源。 點光源從一個點發出光，所有方向的輻射具有相同的顏色和幅度。

- directional light sources
    我們通常使用的第三個假設是某些光源可以建模為定向光源。 在這種情況下，來自光源的光方向在所有 3D 空間位置處都是相同的。 對於光源遠離渲染對象（例如陽光）的情況，定向照明是一個很好的近似模型。

### 3d rendering 
在本節中，我們將了解使用 PyTorch3D 渲染網格模型的具體編碼練習。 我們將學習如何定義相機模型以及如何在 PyTorch3D 中定義光源。 我們還將學習如何更改入射光分量和材質屬性，以便通過控制三個光分量（環境光、漫射光和光澤）來渲染更真實的圖像：


####  Coding exercise for 3D rendering 
> render_3d.py


## Using pytorch3d heterogeneous batches and pytorch optimizers 
學習怎麼用pytorch3d的optimizer在異質小批量

### coding for a heterogenous in mini-batch
來學習如何使用 PyTorch 優化器和 PyTorch3D 異構小批量功能。
在此示例中，我們將考慮深度相機放置在未知位置的問題，並且我們希望使用相機的感測結果來估計未知位置。 為了簡化問題，我們假設相機的方向已知，唯一未知的是 3D 位移。

更具體地說，我們假設相機觀察場景中的三個對象，並且我們知道這三個對象的地面實況網格模型。 讓我們看看使用PyTorch和PyTorch3D解決問題的代碼如下：
> heterogenous_mini_batch.py
## Understanding transformations and rotations
在 3D 深度學習和計算機視覺中，我們通常需要處理 3D 變換，例如旋轉和 3D 剛性運動。 
PyTorch3D 在其 pytorch3d.transforms.Transform3d 類中提供了這些轉換的高級封裝。 
Transform3d 類的優點之一是它是基於小批量的。 
因此，正如 3D 深度學習中經常需要的那樣，可以僅在幾行代碼內對小批量網格應用小批量轉換。 Transform3d的另一個優點是梯度反向傳播可以直接通過Transform3d。
### coding exercise for transformation and rotation
> transformation_rotation.py