# Modeling the Human Body in 3D 
人體姿態估計是人體建模的基石問題。
> 這個章節用python2.7 ？ 

## Formulating the 3D modeling problem 
此建模問題的目標是創建逼真的動畫人體。 更重要的是，這應表現出真實的身體形狀，並且必須根據身體姿勢的變化自然變形並捕捉軟組織運動。

我們將以 3D 方式對人體表面和形狀進行建模，作為對整個人體進行建模的代理
我們不需要模型精確，我們只需要它具有真實的外觀。這使得問題更容易解決.
### defining a good representation 
目標是用低維表示來準確地表示人體。 聯合模型（Joint model）是低維度表示,但是會缺失關於人體的形狀，材質等信息。另一方面，我們可以考慮 oxel 網格表示。 這可以對 3D 身體形狀和紋理進行建模，但它的維度極高，並且自然不適合對由於姿勢變化而產生的身體動力學進行建模。

因此，我們需要一種能夠共同表示身體關節和表面的表示，其中包含有關身體體積的信息。 曲面有多種候選表示； 一種這樣的表示是頂點網格。 蒙皮多人線性 (SMPL) 模型使用此表示法。 一旦指定，該頂點網格將描述人體的 3D 形狀

## Understanding the Linear blend skinning technique 
一旦我們有了 3D 人體的良好表示，我們就想要對其在不同姿勢下的外觀進行建模。 這對於角色動畫尤其重要。這個想法是，蒙皮涉及用皮膚（表面）包裹底層骨架，傳達動畫物件的外觀。 這是動畫行業中使用的術語。 通常，這種表示形式採用頂點的形式，然後使用頂點來定義連接的多邊形以形成表面



## Understanding the SMPL model
正如 SMPL 的縮寫所示，這是一個學習線性模型，根據數千人的資料進行訓練。 該模型基於線性混合蒙皮模的概念建構。 這是一個無人監管的產生模型，使用提供的輸入參數產生 20,670 維向量
我們可以控制。 此模型計算產生正確變形所需的混合形狀
對於不同的輸入參數。 我們需要這些輸入參數具有以下重要屬性：
- 它應該對應於人體真實的有形屬性。
- 這些特徵本質上必須是低維度的。 這將使我們能夠輕鬆控制生成過程。
- 這些特徵必須以可預測的方式被解開和控制。 也就是說，改變一個參數不應改變歸因於其他參數的輸出特性。
  
最重要的是，SMPL 模型是變形的加性模型。 即透過對原始模板體形向量添加形變得到期望的輸出體形向量。 這種附加屬性使得該模型非常易於理解和最佳化

### Defining the SMPL model 
SMPL 模型建立在標準蒙皮模型之上。 它對其進行了以下更改：
- 它不使用標準的休息姿勢模板，而是使用作為身體形狀和姿勢函數的模板網格
- 關節位置是身體形狀的函數
可以用以下的公式來表示
$$
M(\overrightarrow{\beta} ,\overrightarrow{\theta}) = W(T_{p}(\overrightarrow{\beta},\overrightarrow{\theta})\jmath(\overrightarrow{\beta}),\overrightarrow{\theta},W)
$$
其中：
- ß是表示物體特徵（也稱為形狀）的向量。 稍後我們將詳細了解它代表什麼。
- ø是姿勢參數，對應目標姿勢
- W 是線性混合蒙皮模型的混合權重。


該函數看起來與線性混合蒙皮模型非常相似。 在這個函數中，模板網格是形狀和姿態參數的函數，關節的位置是形狀參數的函數。
線性混合蒙皮模型中的情況並非如此。

## Using the SMPL model 
> render_smpl.py
安裝opendr
```
export DISABLE_BCOLZ_AVX2=true
pip install opendr-toolkit-engine
pip install opendr-toolkit
```
## Estimating 3D human pose and shape using SMPLify
從 2D 影像估計 3D 形狀並不總是沒有錯誤。 這是一個具有挑戰性的問題，因為人體、關節、遮蔽、服裝、照明的複雜性，以及從 2D 推斷 3D 時固有的模糊性（因為多個 3D 姿勢在投影時可以具有相同的 2D 姿勢）。


馬克斯普朗克研究所的研究人員發明了實現這一目標的最佳方法之一。
智慧系統（SMPL 模型的發明者）、微軟、馬裡蘭大學和蒂賓根大學。 這種方法稱為 SMPLify。 讓我們更詳細地探討這種方法。
SMPLify 包含兩個步驟
1. 使用已建立的姿勢偵測模型（例如 OpenPose 或 DeepCut）自動偵測 2D 關節。 任何 2D 關節偵測器都可以代替它們使用，只要它們預測相同的關節即可。
2. 使用 SMPL 模型產生 3D 形狀。 直接優化SMPL的參數,使SMPL模型的模型關節投影到前一階段預測的二維關節上

我們知道 SMPL 僅捕捉關節的形狀。 因此，透過 SMPL 模型，我們可以僅從關節捕獲有關身體形狀的信息。 在SMPL模型中，體形參數以β來表徵。 它們是 PCA 形狀模型中主成分的係數。

此姿勢透過運動樹中 23 個關節的相對旋轉和 theta 進行參數化。 我們需要擬合這些參數 β 和 theta，以便最小化目標函數。

### Defining the optimization objective functino
任何目標函數都必須捕捉我們的意圖，以盡量減少某些錯誤的概念。 此誤差計算越準確，優化步驟的輸出就越準確。 我們將首先查看整個目標函數，然後查看該函數的每個單獨組件並解釋為什麼每個組件都是必要的：
$$
E_{j}(\beta,\theta,K,J_{est}) +  \lambda_{\theta} E_{\theta}(\theta) + \lambda_{\alpha}E_{\alpha}(\theta) + \lambda_{sp}E_{sp}(\theta;\beta) + \lambda_{\beta}E_{\beta}(\beta)
$$
我們希望透過最佳化參數 β 和 Ɵ 來最小化該目標函數。 它由四個項目和相應的係數 λƟ、λa、λsp 和 λβ 組成，它們是優化過程的超參數。 

## Exploring SMPLify
> 20231002 有趣可以詳細看看，學習

參考的是這篇論文的實現**Keep it SMPL: Automatic Estimation  of 3D Human Pose and Shape from a Single Image**


### Runing the code 
> run_fit3d.py 

