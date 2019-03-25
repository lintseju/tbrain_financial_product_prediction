玉山人工智慧公開挑戰賽#1 - 金融商品交易預測
======================================

[官方網站](https://tbrain.trendmicro.com.tw/Competitions/Details/5)

第八名作法 (8th/1121, top 1%)

```
# 產生 feature 檔
python feature.py
# 訓練模型及產生上傳檔案
python main.py --online
```

根據比賽規則，比賽資料不得公開。

Problem
========
給顧客基本資料、玉山銀行網站瀏覽紀錄、金融商品交易紀錄，預測顧客是否會在未來三十
天內 1. 購買外匯 2. 申請信用卡 3. 申購信託商品 4. 申請信用貸款。計分方式是計
算各項 f1 score ，然後外匯 x1 ，信用卡 x10 ，信託 x20 ，信貸 x20 後相加，
所以滿分是 51 分。

Offline evaluation
==================
raining data 最後保留三十天做為 validation set 計算 f1 score。

Feature
=======
主要使用兩類 feature
1. 過去 x 天內是否有過交易。
2. 過去 y 天內看過哪個網頁幾次。

不同的 label 有不同的 x, y 。

每個 label 用的網頁不同，以購買外匯為例，先計算看過這個網頁的人有多少比例之後三十
天內有購買外匯作為該網頁的分數，再丟掉看過得人數小於等於十人的。最後取分數最高前 10%
的網頁作為 feature。

Model
=====
LightGBM single model，用 3 fold cross validation (random split) 調
參數。一個 label 訓練一個模型。

Data augmentation
=================
因為瀏覽紀錄只用了最近三十天，訓練資料有提供 120 天，所以把訓練資料分成前 60 天跟
後 60 天建了兩個 dataset 來訓練模型。

Things I missed
===============
1. 所有 testing 出現的顧客都在顧客基本資料裡面有出現，但是有瀏覽紀錄的顧客很多沒
有基本資料，直接使用會有大量缺值。
2. 瀏覽紀錄中，有些活動的網頁在活動結束後就不太會有人去看，可以丟掉。
3. 觀察瀏覽紀錄可以從活動量看出七天一個週期，可以加入星期幾的 feature，但不知道有
沒有用。
4. 嘗試不同 threshold 來最佳化 f1 score。
