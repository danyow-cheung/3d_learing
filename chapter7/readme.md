# Exploring Controllable Neural Feature Fields 

## Understanding GAN-based image synthesis 
圖片生成用深度對抗網路（GAN）
GAN 可以經過訓練，從任何資料分佈中產生外觀相似的影像。 模型StyleGAN2在汽車資料集上進行訓練時，可以產生高解析度的汽車影像：

合成高解析度真實感影像很棒，但這並不是生成模型唯一理想的屬性。 如果生成過程能夠以簡單且可預測的方式解開並可控，則會出現更多現實生活中的應用程式。

更重要的是，我們需要盡可能地分離物體形狀、大小和姿勢等屬性，以便我們可以在不改變圖像中其他屬性的情況下改變它們

現有的基於 GAN 的圖像生成方法在沒有真正理解的情況下生成 2D 圖像
影像的底層 3D 性質。 因此，沒有針對不同屬性（例如物件位置、形狀、大小和姿勢）的內建明確控制。

這會導致 GAN 具有糾纏屬性。 為簡單起見，請考慮一個產生真實臉孔的 GAN 模型的範例，其中改變頭部姿勢也會改變生成的臉孔的感知性別。 如果性別和頭部姿勢屬性糾纏在一起，就會發生這種情況。 對於大多數實際用例來說，這是不想要的。 我們需要能夠改變一個屬性而不影響任何其他屬性。

在下一節中，我們將查看一個模型的高級概述，該模型可以產生 2D 影像，並隱式了解底層場景的 3D 性質。

## Introducing compositional 3D-aware image synthesis 

## Generating feature fields 

## Mapping features fields to images 

## exploring controllable scene generation
### Exploring contraollable car generation 

### Exploring controllable face generation 

## Training the GIRAFFE model 

### Frechet Inception Distance 

### Training the model 

