# Mesh R-CNN
Mask R-CNN 是一種物件偵測和實例分割演算法，在基準資料集中獲得了最高的精確度分數。
它屬於 R-CNN 家族，是一種兩階段端到端目標偵測模型。

Mesh R-CNN 超越了 2D 物件偵測問題，也輸出偵測到的物件的 3D 網格。 如果我們思考世界，人們會看到 3D，這意味著物體是 3D 的。 那麼，為什麼不擁有一個也能輸出 3D 物件的偵測模型呢？

##  Overview of meshes and voxels
Mesh R-CNN 都能使用體素和網格3D表現。

Mesh R-CNN 使用兩種類型的 3D 資料表示。 實驗表明，預測體素然後將其轉換為網格，然後細化網格，有助於網路更好地學習。

## Mesh R-CNN architecture
Mask R-CNN作者的目標是創建一個可端到端訓練的模型。 這就是為什麼他們採用最先進的 Mask R-CNN 模型並添加了一個新的網格預測分支。

<img src = '../imgs/Mesh R-CNN.png'>
Mask R-CNN 的這種結構也與 Mesh R-CNN 相同。 然而，最終添加了網格預測器。 網格預測器是一個新模組，由兩個分支組成：體素分支和網格細化分支。

體素分支將建議的和對齊的特徵作為輸入並輸出粗略的體素預測。

然後將它們作為網格細化分支的輸入，該分支輸出最終的網格。 體素分支和網格細化分支的損失被添加到框架和掩模損失中，並且模型被端到端地訓練：
### Graph convolutions 
> 圖卷積

神經網路的早期變異被用於結構化歐幾里德資料。 然而，在現實世界中，大多數數據都是非歐幾裡得的並且具有圖形結構。 最近，神經網路的許多變體也開始適應圖數據，其中之一是卷積網絡，稱為圖卷積網絡graph convolutional networks（GCN）。

網格有這種圖結構所以圖神經網路適合用在3D預測應用中。


### Mesh predictor 
網格預測器模組旨在檢測物件的 3D 結構。 它是RoIAlign模組的邏輯延續，負責預測和輸出最終網格。

當我們從現實生活影像中取得 3D 網格時，我們不能使用具有固定網格拓撲的固定網格模板。
這就是網格預測器由兩個分支組成的原因。 體素分支和網格細化分支的組合有助於減少固定拓樸的問題。

體素損失(voxel loss)是二元交叉熵，它最小化體素佔用與地面真實佔用的預測機率。

圖卷積採用影像對齊特徵並沿網格邊緣傳播訊息。 頂點細化更新頂點位置。 它的目的是透過保持拓樸固定來更新頂點幾何形狀：

## Demo of Mesh R-CNN with PyTorch
repo: https://github.com/facebookresearch/meshrcnn.git

```
python demo/demo.py \
--config-file configs/pix3d/meshrcnn_R50_FPN.yaml \
--input /path/to/image \
--output output_demo \
--onlyhighest MODEL.WEIGHTS meshrcnn://meshrcnn_R50.pth
```

render images from the 3D object
> viz_demo_result.py 