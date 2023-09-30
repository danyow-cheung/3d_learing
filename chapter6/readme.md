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
    
)
```
## Understanding volume rendering with radiance fields
