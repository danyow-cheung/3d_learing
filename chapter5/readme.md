# Understanding Differentiable Volumetric Rendering 
了解體素的渲染，我需要了解幾個重要的概念比如說
1. ray  sampling volumes射線採樣體積
2. volume sampling 體積採樣
3. ray marching 雷行進

體素渲染是可微的，所以可以使用不同體素去渲染不同的應用。
## Overview of volumetric rendering 
體積渲染是用於產生離散 3D 資料的 2D 視圖的技術的集合。

當生成表面困難或可能導致錯誤時，通常會使用體積渲染。

## Understanding ray sampling 
射線採樣是從相機發射射線的過程，射線穿過影像像素並沿著這些射線採樣點。

下面，我們將示範如何使用 PyTorch3D 射線採樣器之一 NDCGridRaysampler。 這就像 NDCMultiNomialRaysampler，其中像素沿著網格採樣。 這些程式碼可以在名為 Understanding_ray_sampling.py 

## Using volume sampling 
體積採樣是沿著光​​線樣本提供的點獲取顏色和佔用資訊的過程。

## Exploring the ray number 
現在我們有了光線採樣器採樣的所有點的顏色和密度值，我們需要弄清楚如何使用它來最終渲染投影影像上的像素值。 在本節中，我們將討論將光線點上的密度和顏色轉換為影像上的 RGB 值的過程。 該過程模擬了影像形成的物理過程。


在本節中，我們討論一個非常簡單的模型，其中每個影像像素的 RGB 值是相應光線點上顏色的加權和。 如果我們將密度視為佔用或不透明度的機率，則光線每個點的入射光強度是 a = (1-p_i) 的乘積，其中 p_i 是密度。 假設該點被某物體佔據的機率為p_i，則從該點反射的預期光強度為w_i = a p_i。 我們只使用 w_i 作為顏色加權和的權重。 通常，我們透過應用 softmax 運算來標準化權重，使權重總和為 1。

## Differentiable volumetric rendering 
標準體積渲染用於渲染 3D 資料的 2D 投影，而可微分體積渲染則用於執行相反的操作：從 2D 影像建構 3D 資料。 它的工作原理是這樣的：我們將物件的形狀和紋理表示為參數函數。 此函數可用於產生 2D 投影。 但是，給定 2D 投影（這通常是 3D 場景的多個視圖），我們可以優化這些隱式形狀和紋理函數的參數，使其投影是多視圖 2D 影像。 這種最佳化是可能的，因為渲染過程是完全可微的，並且使用的隱式函數也是可微的。

## Summary
