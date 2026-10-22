# 藉由 NOMAD 損失與可懂度損失以改進語音強化之 MP-SENet 深度模型其效能

**Leveraging the NOMAD and Intelligibility Loss to Improve MP-SENet for Speech Enhancement**

- **學生姓名**：李宜蓁、張中瑋
- **指導教授**：洪志偉老師
- **單位**：國立暨南國際大學 電機工程學系（Department of Electrical Engineering, National Chi Nan University）

---

## 摘要與主要貢獻

### 本專題目的與主軸

1. 研究與實現**語音強化法**，消除環境雜訊對語音的干擾，從而增強語音訊號的品質。
2. 藉由 Python 程式的撰寫與深度學習理論的研究、探討 **MP-SENet 語音強化法**，**改變損失函數**以提升模型之效能。

### 我們提出的新方法

直接修改 MP-SENet 上的模型損失函數，包括了：

1. 將 **NOMAD loss** 加入到**生成器**損失中
2. 將 **STOI loss** 加入到**鑑別器**損失中
3. 探討生成器各項損失對模型效能的影響

### 評估實驗的主要發現

所提出的新方法皆可提升 MP-SENet 在語音強化之各項指標分數（PESQ、CSIG、CBAK、COVL、SSNR、STOI）。

### 研究成果發表

本研究專題之相關研究成果，已投稿至二個國際會議與一個國內會議：

- **IEEE ICASI 2024**（已被接受並發表）— IEEE International Conference on Applied System Innovation 2024
- **IEEE ICCE-TW 2024**（已被接受並將在下半年發表）— IEEE International Conference On Consumer Electronics - Taiwan 2024
- **ICEAT 2024**（尚在審核中）

---

## 簡介

### 語音訊號受雜訊干擾示意圖

乾淨語音 $x(t)$ 加上背景雜訊 $n(t)$（Adaptive noise），再經過通道效應 $h(t)$（Convolution）後，形成含噪語音：

```
Clean x(t) ──(+)── n(t) 背景雜訊 ──(×)── h(t) 通道效應 ──► Noisy
```

### 語音強化的目的

1. 通過去除雜訊和其他干擾因素，提高語音辨識精確率。
2. 提升語音品質，使人耳能接受到更清晰的語音。

---

## MP-SENet 架構

### MP-SENet 基本原理

1. 採用編解碼器架構，透過**編碼器**提取特徵
2. 藉由兩級的**卷積增強轉換器**（Conformer）捕捉時頻依賴性
3. **同時對強度頻譜和相位頻譜去噪**，藉此提高語音訊號品質

### 流程圖組成

- **編碼器（Encoder）**
- **強度頻譜解碼器（Magnitude Mask Decoder）**
- **相位頻譜解碼器（Phase Decoder）**
- 損失項：Magnitude Loss、Metric Loss（Metric Discriminator）、Time Loss、Complex Loss、Anti-wrapping Phase Loss

---

## 對於 MP-SENet 所提的改良法：I & II

### 改良法 I：在生成器損失函數中加入 NOMAD loss

1. NOMAD 是一種**非監督的感知相似度度量**，評估受損訊號與非匹配參考之間的距離。
2. NOMAD 基於使用由 NSIM 監督的 triplet loss 學習特徵嵌入：

$$
\sum_{i}^{N} \max\left(0,\ \lVert f(\boldsymbol{x}_i^a) - f(\boldsymbol{x}_i^p) \rVert_2^2 - \lVert f(\boldsymbol{x}_i^a) - f(\boldsymbol{x}_i^n) \rVert_2^2 + m\right)
$$

3. 透過更進一步考慮清晰度與語音品質，增加語音強化表現。

### 改良法 II：在鑑別器損失函數中加入 STOI loss

1. STOI 是一種用於語音評估的客觀指標，計算增強訊號與乾淨訊號之間的傅立葉轉換頻譜圖中的相關性。
2. 原始鑑別器採用深度神經網絡架構來模擬 PESQ，由於 PESQ 函數不可微分，**透過調整 MP-SENet 鑑別器，並額外考慮 STOI 以估計增強的訊號的可懂性**。

---

## MP-SENet 之改良法 III

### 考慮生成器中所有損失函數對模型效能的影響

1. 從以下修改過後的生成器損失公式可以發現包含了六種不同的損失函數，但**參考越多的損失函數並不代表模型的效能越佳**：

$$
\begin{aligned}
\tilde{\mathcal{L}}_G &= \mathcal{L}_G + \gamma_6 \mathcal{L}_{\text{NOMAD}} \\
&= \gamma_1 \mathcal{L}_{\text{Time}} + \gamma_2 \mathcal{L}_{\text{mag}} + \gamma_3 \mathcal{L}_{\text{Com.}} + \gamma_4 \mathcal{L}_{\text{Metric}} + \gamma_5 \mathcal{L}_{\text{Pha}} + \gamma_6 \mathcal{L}_{\text{NOMAD}}
\end{aligned}
$$

2. 我們透過**減少生成器中的損失函數**來驗證是否參考所有損失函數可達模型的最佳效果。

---

## 實驗環境設置

### 語音資料

從 VoiceBank-DEMAND 語音資料庫中選取 11572 個音檔做為訓練集、824 個音檔作為測試集。

### 語音強化評估指標

| 指標 | 分數範圍 | 說明 |
|------|----------|------|
| PESQ | [-0.5, 4.5] | 分數越高表示語音品質越好 |
| STOI | [0, 1] | 分數越接近 1 表示語音清晰度越高 |
| SI-SNR | — | 分數越高，表示語音增強效果越好 |
| CSIG | [1, 5] | 分數越高表示語音失真程度越低 |
| COVL | [1, 5] | 分數越高表示語音整體品質越好 |
| CBAK | [1, 5] | 分數越高表示背景失真程度越低 |

---

## MP-SENet 及更改損失函數之時頻圖

海報比較四張頻譜圖：(1) Noisy、(2) Clean、(3) Enhanced by MP-SENet、(4) Enhanced by revised MP-SENet。

> 相較於 (3)，新方法 (4) 得到更乾淨的頻譜圖。

---

## 實驗數據

### 表一、STOI 損失和不同權重的 NOMAD 損失對 MP-SENet 的改善結果

| 方法 | PESQ | CSIG | CBAK | COVL | SSNR | STOI (%) |
|------|------|------|------|------|------|----------|
| MP-SENet baseline ($\gamma_6 = 0$) | 3.33 | 4.63 | 3.86 | 4.08 | 10.76 | 95.60 |
| Revised MP-SENet with NOMAD loss, $\gamma_6 = 0.01$ | 3.35 | 4.66 | **3.87** | 4.10 | 10.82 | **95.74** |
| Revised MP-SENet with NOMAD loss, $\gamma_6 = 0.05$ | 3.31 | 4.64 | 3.84 | 4.07 | 10.69 | 95.67 |
| Revised MP-SENet with NOMAD loss, $\gamma_6 = 0.1$ | 3.33 | 4.65 | 3.86 | 4.08 | 10.80 | 95.68 |
| Revised MP-SENet with NOMAD loss, $\gamma_6 = 0.5$ | **3.36** | **4.67** | 3.86 | **4.12** | 10.56 | 95.59 |
| Revised MP-SENet with NOMAD loss, $\gamma_6 = 1.0$ | 3.25 | 4.61 | 3.81 | 4.03 | 10.26 | 95.45 |
| MP-SENet with STOI loss | 3.31 | 4.65 | 3.85 | 4.08 | 10.76 | 95.64 |
| MP-SENet with NOMAD and STOI loss | 3.34 | 4.65 | **3.87** | 4.09 | **10.83** | 95.66 |

（粗體為各指標最佳值）

### 表二、移除各項損失函數對 MP-SENet 的影響

| 方法 | PESQ | CSIG | CBAK | COVL | SSNR | STOI (%) |
|------|------|------|------|------|------|----------|
| MP-SENet baseline | 3.33 | 4.63 | 3.86 | 4.08 | 10.76 | 95.60 |
| With NOMAD loss ($\gamma_6 = 0.01$) | 3.35 | 4.66 | 3.87 | 4.10 | 10.82 | 95.74 |
| With NOMAD loss but removing metric loss ($\gamma_6 = 0.01$) | 3.21 | 4.61 | 3.81 | 4.01 | **10.91** | 95.66 |
| With NOMAD loss but removing time loss ($\gamma_6 = 0.01$) | 3.34 | 4.66 | 3.82 | 4.10 | 10.05 | 95.61 |
| With NOMAD loss but removing complex loss ($\gamma_6 = 0.01$) | 3.31 | 4.65 | 3.85 | 4.08 | 10.68 | 95.69 |

---

## 實驗結果與討論

### 實驗數據分析

- 觀察表一，**使用 $\gamma_6 = 0.01$ 的 NOMAD 損失時，在所有使用的 SE 指標中均優於原始 MP-SENet**。
- 其他 $\gamma_6$ 之設定也能提升部分語音強化指標，顯示 NOMAD 損失的加入能有效提升 MP-SENet。
- 將 STOI 損失整合到鑑別器的損失函數中，相對於 MP-SENet 基準，提高了 CSIG、SSNR 和 STOI 指標。STOI 的增加幅度很小，可能的原因是 STOI 分數已經接近其最優值 1，且很難進一步提高。
- 同時考慮 NOMAD 和 STOI 損失對 MP-SENet 的性能幾乎在所有 SE 指標上優於 MP-SENet，而在 SSNR 指標上優於由 NOMAD 改善的 MP-SENet。
- 由於 MP-SENet 是針對強度及相位頻譜進行去噪，因此我們保留生成器中的相位及幅度損失，並單一去除剩餘的時域損失、複數損失和度量損失，此舉是希望透過 NOMAD 損失分別取代以上三種損失以降低生成器損失函數總量。
- 觀察表二，當移除特定損失時，部分指標呈上升趨勢：**當去除度量損失時，SSNR 達到最高值 10.91，相較於 baseline 提升了 0.15**，而 STOI 指標分數則是在移除特定損失時都有所進步。

### 曲線圖分析

海報附四張曲線圖：不同損失對 MP-SENet 的 SSNR／STOI 損益改善結果，以及移除各項損失對 MP-SENet 的 SSNR／STOI 影響。

1. 與原始 MP-SENet 相比，**大部分加入 NOMAD 與 STOI 損失改善的權重都使模型在 SSNR 和 STOI 分數表現上更好**，然而過高的 NOMAD 損失函數權重會導致效能降低，導致語音強化指標分數上的不理想。
2. 在分別移除各項損失時，**STOI 分數都有上升**，顯示我們加入的 NOMAD 損失在改善 STOI 上有適度的成效，在移除 Metric 損失後語音增強效果最好（SSNR 指標分數最高）。

### 結論

在這項研究中，我們提出**將 STOI 損失整合到鑑別器中，將 NOMAD 損失整合到生成器中來修改 MP-SENet**。所提出的修改在 VoiceBank-DEMAND 數據集和任務上對 MP-SENet 的各種語音強化指標帶來了適度的改善。在未來研究中，我們將繼續專注於通過 NOMAD 損失和 STOI 損失的幫助，減少 MP-SENet 架構的參數，同時保持出色的性能。
