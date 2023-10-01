# Exploring Neural Radiance Fields
神經輻射場

使用神經輻射場來表達3d信息

## Understanding NeRF
視圖合成（View synthesis ）是 3D 電腦視覺中長期存在的問題。 挑戰在於合成新的使用場景的少量可用 2D 快照來查看 3D 場景。 它特別是具有挑戰性，因為複雜場景的視圖可能取決於許多因素，例如物件偽影、光源、反射、不透明度、物體表面紋理和遮蔽。

NeRF 使用非常規的方式來使用神經網絡

在本節中，首先，我們將探討輻射場(radiance fields)的意義以及如何使用神經網路來表示這些輻射場。

###  What is a radiance field?
輻射度(Radiance)是測量穿過特定立體角內的區域或從特定立體角內的區域發出的光量的標準指標

就我們的目的而言，我們可以將輻射率視為
從特定方向觀察時空間中一點的強度。 以 RGB 形式捕捉此資訊時，亮度將具有與紅色、綠色和藍色相對應的三個分量。 空間中一點的輻射率可能取決於許多因素，包括以下因素
- Light sources illuminating that point 光源照亮該點
- The existence of a surface (or volume density ) that can reflect light at that point 存在可以在該點反射光的表面（或體積密度）
- The texture properties of the surface 表面的紋理屬性 
  
輻射場只是 3D 場景中所有點和視角的輻射值的集合：
`輻射（r,g,b）在點（x,y,z)的特定角度(θ,ø)`
### Representing radiance fields with neural networks 
NeRF 使用神經網路來表示體積場景函數。 此神經網路採用 5 維輸入,具體輸入是`(x,y,z,θ,ø)`
Its output is the volume density σ at (x, y, z) and the emitted color (r, g, b) of the point (x, y, z) when
viewed from the viewing angle (θ, ∅). 
其輸出為 (x, y, z) 處的體積密度 σ 以及從視角 (θ, ∅) 觀看時 (x, y, z) 點的發射顏色 (r, g, b)。

## Training a NeRF model 
> chapter6_trainNeRF.ipynb 


## Understanding the NeRF model architecture 
神經網路採用空間位置 (x, y, z) 的諧波(harmonic)嵌入和諧波嵌入 (θ, ∅) 作為其輸入並輸出預測密度 σ 和預測顏色 (r, g, b)。

每个输入点都是一个 5 维向量。 发现直接在这个上训练模型
当表示颜色和几何形状的高频变化时，输入表现不佳。 这
是因为众所周知，神经网络偏向于学习低频函数。
解决这个问题的一个好方法是将输入空间映射到更高维的空间并使用它进行训练。 该映射函数是一组具有固定但唯一频率的正弦函数：
$$
\gamma(p) = (sin(2^0 \pi p),cos(2^0 \pi p),...sin(2^{L-1} \pi p ),cos(2^{L-1} \pi p ))
$$
该函数应用于输入向量的每个分量：
> chapter6_NeRF_architute.py
```
import torch 
class NeuralRadianceField(torch.nn.Module):
    def __init__(self,n_harmonic_functions=60,n_hidden_neurons=256) -> None:
        super().__init__()
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)

```
神经网络由 MLP 主干组成。 它采用位置 (x, y, z) 的嵌入
作为其输入。 这是一个全连接网络，使用的激活函数是softplus。
softplus 函数是 ReLU 激活函数的平滑版本。 输出
主干的向量是一个大小为 n_hidden_neurons 的向量：
```
embedding_dim = n_harmonic_functions*2*3 
self.mlp = torch.nn.Sequential(
    torch.nn.Linear(embedding_dim,n_hidden_neurons),
    torch.nn.Softplus(beta=1.0),
    torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
    torch.nn.Softplus(beta=10.0),
)
```
我们定义了一个颜色层，它采用 MLP 主干的输出嵌入以及
光线方向输入嵌入并输出输入的 RGB 颜色。 我们将这些结合起来输入，因为颜色输出强烈依赖于点的位置和
观察方向，因此，提供较短的路径来利用这一点非常重要
神经网络：
```
self.color_layer = torch.nn.Sequential(
    torch.nn.Linear(n_hidden_neurons+embedding_dim,n_hidden_neurons),
    torch.nn.Softplus(beta=10.0),
    torch.nn.Linear(n_hidden_neurons,3),
    torch.nn.Sigmoid(),
)
```
5. 接下来，我们定义密度层。 点的密度只是其位置的函数：

```
self.density_layer = torch.nn.Sequential(
    torch.nn.Linear(n_hidden_neurons,1),
    torch.nn.Softplus(beta=10.0),
    self.density_layer[0].bias.data[0] = -1.5
)
```
6. 现在，我们需要一些函数来获取 Density_layer 的输出并预测原始密度：
```python
def _get_densities(self,features):
    raw_densities = self.density_layer(features)
    return 1- (-raw_densities).exp()

```
7. 我们在给定光线方向的特定点上执行相同的操作来获取颜色。 我们需要首先将位置编码函数应用于射线方向输入。 然后我们应该将它与 MLP 主干的输出连接起来：
``` python 
def _get_colors(self,features,rays_directions):
    spatial_size = features.shape[:-1]
    rays_directions_normed = torch.nn.functional.normalize(
        rays_directions,dim=-1
    )
    rays_embedding = self.harmonic_embedding(
        rays_directions_normed
    )
    rays_embedding_expand = rays_embedding[...,None,:].expand(
        *spatial_size,rays_embedding.shape[-1]
    )
    color_layer_input = torch.cat(
        (features,rays_embedding_expand
    ),dim=-1)
    return self.color_layer(color_layer_input)

```
8. 我们定义前向传播的函数。 首先，我们获得嵌入。 然后，我们将它们传递给 MLP 主干以获得一组特征。 然后我们用它来获得密度。 我们使用特征和光线方向来获取颜色。 我们返回密度和颜色：
```python
def forward(self,ray_bundle,RayBundle,**kwargs):
    rays_points_world = ray_bundle_to_gray_points(ray_bundle)
    embeds = self.harmonic_embedding(rays_points_world)
    features = self.mlp(embeds)
    rays_densities = self._get_densities(features)
    rays_colors = self._get_colors(features,ray_bundle.directions)
    return rays_densities,rays_colors
```
9. 该函数用于对输入光线进行内存高效的处理。 首先，输入光线被分成 n_batches 块，并在 for 循环中一次一个地通过 self.forward 函数。 与禁用 PyTorch 梯度缓存 (torch.no_grad()) 相结合，这使我们能够在一次前向传递中渲染不完全适合 GPU 内存的大批量光线。 在我们的例子中，batched_forward 用于导出辐射场的全尺寸渲染以用于可视化目的：
```python
def batched_forward(self,ray_bundle:RayBundle,n_batches:int = 16,**kwargs):
    n_pts_per_ray = ray_bundle.lengths.shape[-1]
    spatial_size = [*ray_bundle.origins.shape[:-1],n_pts_per_ray]
    # split the rays to `n_batches` batches 
    tot_samples = ray_bundle.origins.shape[:-1].numel()
    batches = torch.chunk(torch.arange(tot_samples),n_batches)
```
10. 对于每个批次，我们需要首先运行前向传递，然后分别提取 ray_densities 和 ray_colors 以作为输出返回：
```python 
batch_outputs = [self.forward(
    RayBundle(
        origins=ray_bundle.origins.view(-1,3)[batch_idx],
        directions = ray_bundle.directions.view(-1,3)[batch_idx],
        lengths = ray_bundle.lengths.view(-1,n_pts_per_ray)[batch_idx],
        xys=None,
    )
)    for batch_idx in batches 
]

rays_densities ,rays_colors  = [
    torch.cat(
        [batch_output[output_i] for batch_output in batch_outputs],dim=0
    ).view(*spatial_size,-1)for output_i in (0,1)]
return rays_densities,rays_colors
```

## Understanding volume rendering with radiance fields
体积渲染允许您创建 3D 图像或场景的 2D 投影。

### Projecting rays into the scene
想象一下将相机放置在一个视点并将其指向感兴趣的 3D 场景。 这是NeRF 模型训练的场景。 为了合成场景的 2D 投影，我们首先发送光线从视点射入 3D 场景。

射线可以参数化如下：
$$
r(t) = o + td 
$$
这里，r是从原点o出发沿d方向传播的射线。 它是参数化的t，可以改变它以移动到射线上的不同点。 请注意，r 是 3D 向量,代表空间中的一个点。

### Accumulating the color of a ray 
