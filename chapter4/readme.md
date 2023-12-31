# Learning object pose detection and tracking by differentiable rendering 
為了有效地解決上述優化問題，渲染過程應該是可微的。 例如，如果渲染是可微的，我們可以使用端到端的方法來訓練深度學習模型來解決問題。 然而，正如後面幾節將更詳細討論的，傳統的渲染過程是不可微分的。 因此，我們需要修改傳統方法以使它們可微分。 我們將在下一節中詳細討論如何做到這一點。

## Why we want to hace differentiable rendering 
物理過程是把3d模型影射到二維圖像中

根據幾十年前計算機視覺社區首次討論的一些想法，我們可以將該問題表述為優化問題。 在本例中，此處的優化變量是兩個 3D 球體的位置。 我們想要優化這兩個中心，使得渲染的圖像就像前面的 2D 觀察到的圖像一樣。 為了精確測量相似性，我們需要使用成本函數——例如，我們可以使用像素級均方誤差。 然後，我們需要計算從成本函數到兩個球體中心的梯度，以便我們可以通過朝梯度下降方向迭代地最小化成本函數。


## How to make rendering differentiable 
渲染是對圖像形成的物理過程的模仿。 在許多情況下，圖像形成本身的物理過程是可微的。 假設表面是法線且物體的材質屬性都是光滑的。 那麼，示例中的像素顏色是球體位置的可微函數。


當我們使用傳統的渲染算法時，由於離散化，有關局部梯度的信息會丟失。 正如我們在前面的章節中討論的，光柵化是渲染的一個步驟，對於成像平面上的每個像素，我們找到最相關的網格面（或者決定找不到相關的網格面）。


### What problems can be solved by using differentiable rendering 
如前所述，可微渲染在計算機視覺社區中已經討論了數十年。 過去，可微渲染用於單視圖網格重建、基於圖像的形狀擬合等。 在本章的以下部分中，我們將展示使用可微渲染進行剛性物體姿態估計和跟踪的具體示例。
可微渲染是一種可以將 3D 計算機視覺中的估計問題轉化為優化問題的技術。 它可以應用於廣泛的問題。 更有趣的是，最近一個令人興奮的趨勢是將可微分渲染與深度學習相結合。 通常，可微渲染被用作深度學習模型的生成器部分。 因此，整個管道可以進行端到端的訓練。

## The object post estimation problem
the problem is object post estimation from one single observed image 
> diff_render.py
## how it is work 

### An example of object pose estimation for both silhouette fitting and texture fitting 
在前面的例子中，我們通過輪廓擬合來估計物體姿態。 在本節中，我們將介紹使用輪廓擬合和紋理擬合進行物體姿態估計的另一個示例。 在3D計算機視覺中，我們通常使用紋理來表示顏色。 因此，在這個例子中，我們將使用可微分渲染根據相機位置來渲染RGB圖像並優化相機位置。 代碼在 diff_render_texture.py 中：
> diff_render_texture.py