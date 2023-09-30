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

## Understanding the NeRF model architecture 
## Understanding volume rendering with radiance fields
