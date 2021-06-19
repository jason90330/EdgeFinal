## 1. 作品名稱: `顏語情`

## 2. 摘要說明
搭載 Rgb sensor 以及 Thermal sensor 擷取 Rgb image 以及 Thermal image ，並使用 Retina face detector Crop 出影像臉部區域， 最後訓練一個較小型的 cnn model，如 efficientnet-b0 來預測使用者的情緒。


## 3. 系統簡介
### 3-1. 創作發想
目前業界當中有許多電影公司在試片時都需要記錄使用者的情緒，但是比較可惜的是影片通常都很長，觀眾無法紀錄每一刻的想法以及情緒，最後僅能透過口頭上的解釋對於這部片的想法，但製片商以及導演最需要的仍然是每個畫面是否能夠為觀眾帶來衝擊，因此這個作品的創作出發點也是希望能夠記錄住觀眾每刻的情感狀態，進而達到電影影評分析的目的，另外為了要能夠滿足某些試片場景僅有微弱的光通量，所以這邊我就考慮到可以使用 Thermal camera ，依靠 IR images train 出來的 model 來辨識黑暗中使用者的情緒。

### 3-2. 硬體架構
* 硬體: Kinect
![](https://i.imgur.com/RPkFzAc.png)
    ```
    RGB Camera:
        解析度: 1920 * 1080
        FPS: 30
    Depth Camera: 
        解析度: 512*424
        FPS: 30
        感測距離: 0.5m ~ 4m
    ```
* 原理
發射的近紅外線光是依據 TOF(Time of Flight) 技術來量測光線飛行時間，所以愈遠的地方會愈暗(也代表深度愈深)


### 3-3. 工作原理及流程
使用 Kinect IR Camera 作為感測器，因為會有遠近量案的差異，所以有根據 Depth 用線性的方式針對較遠的像素呈上一個倍率加強明亮度，資料做完前處理後便使用 Retina face detector 抓取人臉，並做裁切，最後在餵給模型，並針對不同的情緒類別先做一開始的機率初始化讓受到 IR 影像與 RGB 影像所導致的類別不平衡的狀況消失。
![](https://i.imgur.com/naK7w7n.png)


### 3-4. 資料集建立方式
![](https://i.imgur.com/dIQx4Wv.png)
* **資料集:** AffectNet
* **資料前處理:** 
    1. 將資料去除掉 Uncertain, Non-Face, Non-Exist 後平均每個類別各取 10,000 張
    2. 將一半的圖片轉為灰階(因為要拿去給 Kinect IR 使用，灰階色相較相似)
    3. 正歸化
    4. 縮放為 224 * 224
    5. 隨機水平翻轉

### 3-5. 模型選用與訓練
* **訓練模型:** [EfficientNet](https://arxiv.org/abs/1905.11946)
* **訓練設定:** Warm up 2000 iterations訓練 30 epochs
* **Loss criterion:** LabelSmoothSoftmaxCE
* **Optimizer:** AdamW
* **Scheduler:** ReduceLROnPlateau
![](https://i.imgur.com/HZuPRaQ.png)

## 4. 實驗結果
### 4-1. 測試與比較
實驗比較上跟下面這篇論文做比較，他們所提出的架構與 ResNet-18 類似:
* https://arxiv.org/pdf/2103.10189v1.pdf
2021  Learning to Amend Facial Expression Representation via De-albino and Affinity
![](https://i.imgur.com/cba31OW.png)

而下方為我們的實驗結果:
* Accuracy (8 classes): 56.156% (epoch-11)
![](https://i.imgur.com/JGn9hzr.png)

### 4-2. 改進與優化
使用硬體: Intel I7-9750H
![](https://i.imgur.com/shLd9J8.png)

## 5. 延伸方向
另外這邊也提出了一個新的構想，就是透過現今較為主流的換臉模型將我的 Kinect IR 影片轉換為彩色的影片，背後僅需要提供原始待測 IR 影片以及單張我用 StyleGAN-V2 所生成的無個資人臉，透過這種方法的優點是能夠間接達到 Super-resolution、轉換色彩、保護身分個資目的，但伴隨而來的缺點是，若原先 Face landmark 沒找好可能就表現不好
![](https://i.imgur.com/4VFhAYN.png)

## 6. 參考文獻
[RetinaFace](https://arxiv.org/abs/1905.00641)
[EfficientNet](https://arxiv.org/abs/1905.11946)
[First Order Motion](https://papers.nips.cc/paper/2019/file/31c0b36aef265d9221af80872ceb62f9-Paper.pdf)
[2021 Emotion Paper (ARM)](https://arxiv.org/pdf/2103.10189v1.pdf)


## 7. 附錄
### 7-1. Colab源碼
https://github.com/jason90330/EdgeFinal
### 7-2. 資料集及標註檔
https://drive.google.com/drive/folders/1zx9qlejfSYX7Z-nvklHb9MfOIxDPpYTA?usp=sharing
https://drive.google.com/drive/folders/1wfMw_THSLuv-QaKl9vN4rbQem_x0YdAA?usp=sharing
