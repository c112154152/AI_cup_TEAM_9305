# YOLO-Based Object Detection Evaluation  
NKUST 機器學習期末專題 / AI Cup 競賽實作

本專案為 **NKUST 機器學習課程期末專題暨 AI Cup 競賽實作** 之程式碼整理，核心內容為使用 **Ultralytics YOLO 架構** 進行影像物件偵測模型訓練，並實作一套 **YOLO 預測結果之 IoU 與 TP / FP / FN 評估工具**，用以分析模型在測試集上的實際表現。

---

## 一、專案簡介

本研究以 AI Cup 官方提供之影像資料集為基礎，進行完整的資料前處理、模型訓練、測試集推論與效能評估流程。  
模型訓練與推論皆使用本機環境完成，並以 **YOLOv9-m 預訓練權重** 進行學習。

除官方評估指標外，本專案額外實作 `eval_yolo.py`，針對測試集進行 **IoU、True Positive、False Positive、False Negative 與多餘預測框分析**，以利進一步誤差檢視與模型除錯。

---

## 二、執行環境

- 平台：本機
- 作業系統（測試）：Windows 11（僅環境驗證，未產生成果）
- GPU（Colab）：NVIDIA Tesla T4
- 程式語言：Python
- 主要套件：
  - ultralytics (YOLO)
  - gdown
  - os, shutil, csv, math

---

## 三、資料處理方式

### 3.1 資料來源
- AI Cup 官方提供之影像資料集
- 影像格式：PNG
- 標註格式：YOLO bounding box (`.txt`)

### 3.2 資料切分
- 依 YOLO 建議機制進行資料切分
- 訓練集 : 驗證集 : 測試集 = **5 : 4 : 1**
- 以病人（patient）為單位切分，避免資料洩漏

### 3.3 資料增強
訓練階段採用合理之醫學影像增強策略：
- 水平翻轉
- 垂直翻轉
- 影像縮放
- HSV 色彩空間調整
- 低比例 Mosaic（0.05）

---

## 四、模型訓練設定

- 模型架構：YOLOv9-m
- 預訓練權重：YOLOv9-m pretrained
- Epochs：100 – 150
- Batch size：8
- Image size：640 × 640
- Loss weights：
  - box = 7.5
  - cls = 0.5
  - dfl = 1.5
- Optimizer：Ultralytics 預設自動最佳化設定

---

## 五、專案結構

```text
.
├── train.ipynb               # 模型訓練與推論（Google Colab）
├── eval_yolo.py              # YOLO 預測結果評估程式
├── datasets/
│   └── test/
│       ├── images/           # 測試影像
│       └── labels/           # Ground Truth 標註
├── runs/
│   └── detect/
│       └── val/
│           └── labels/       # YOLO 預測輸出標註
├── eval_result.csv           # 逐 bounding box 評估結果
└── eval_report.txt           # 評估摘要報告
