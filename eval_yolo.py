import os
import csv
import math

TEST_IMG_DIR = "./datasets/test/images"
TEST_LBL_DIR = "./datasets/test/labels"
PRED_DIR = "./runs/detect/val/labels"

OUTPUT_CSV = "./eval_result.csv"
REPORT_FILE = "./eval_report.txt"

IOU_THRESHOLD = 0.5


def load_label_file(path):
    """讀取 YOLO txt：class cx cy w h"""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            boxes.append((cls, cx, cy, w, h))
    return boxes


def yolo_to_xyxy(box, img_w=512, img_h=512):
    _, cx, cy, w, h = box
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return [x1, y1, x2, y2]


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = area1 + area2 - inter_area
    if union == 0:
        return 0
    return inter_area / union


def match_predictions(gt_boxes, pred_boxes, img_name):
    results = []
    matched_pred = set()

    max_iou_for_img = 0
    tp_count = 0

    for gt in gt_boxes:
        gt_xyxy = yolo_to_xyxy(gt)
        best_iou = 0
        best_pred_idx = -1

        for i, pred in enumerate(pred_boxes):
            pred_xyxy = yolo_to_xyxy(pred)
            score = iou(gt_xyxy, pred_xyxy)
            if score > best_iou:
                best_iou = score
                best_pred_idx = i

        max_iou_for_img = max(max_iou_for_img, best_iou)

        if best_iou >= IOU_THRESHOLD:
            tp_count += 1
            matched_pred.add(best_pred_idx)
            results.append([img_name, gt_xyxy, yolo_to_xyxy(pred_boxes[best_pred_idx]), best_iou, "TP"])
        else:
            results.append([img_name, gt_xyxy, None, best_iou, "FN"])

    for i, pred in enumerate(pred_boxes):
        if i not in matched_pred:
            results.append([img_name, None, yolo_to_xyxy(pred), 0, "FP"])

    return results, max_iou_for_img, tp_count


def main():
    print("開始 YOLO 評估...")

    all_rows = []
    img_files = sorted(os.listdir(TEST_IMG_DIR))

    FP_list = []
    FN_list = []
    TP_list = []
    duplicate_list = []

    total_gt = 0
    total_pred = 0
    total_TP = 0
    total_FP = 0
    total_FN = 0
    iou_sum = 0
    iou_count = 0

    for img_name in img_files:
        if not img_name.endswith(".png"):
            continue

        base = img_name.replace(".png", "")
        gt_path = os.path.join(TEST_LBL_DIR, base + ".txt")
        pred_path = os.path.join(PRED_DIR, base + ".txt")

        gt_boxes = load_label_file(gt_path)
        pred_boxes = load_label_file(pred_path)

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        rows, max_iou_for_img, tp_for_img = match_predictions(gt_boxes, pred_boxes, img_name)
        all_rows.extend(rows)

        # collect iou stats
        if max_iou_for_img > 0:
            iou_sum += max_iou_for_img
            iou_count += 1

        # collect error lists
        for row in rows:
            img, gt, pred, iou_score, error = row
            if error == "TP":
                TP_list.append(img)
                total_TP += 1
            elif error == "FP":
                FP_list.append(img)
                total_FP += 1
            elif error == "FN":
                FN_list.append(img)
                total_FN += 1

        # duplicate 检查
        if len(pred_boxes) > len(gt_boxes) and len(gt_boxes) > 0:
            duplicate_list.append(img_name)

    # write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "gt_bbox", "pred_bbox", "IoU", "error_type"])
        writer.writerows(all_rows)

    avg_iou = iou_sum / iou_count if iou_count > 0 else 0

    # write summary report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# YOLO 評估摘要\n\n")
        f.write(f"IoU 門檻: {IOU_THRESHOLD}\n")
        f.write(f"影像數量: {len(img_files)}\n")
        f.write(f"GT 數量: {total_gt}\n")
        f.write(f"預測框數: {total_pred}\n")
        f.write(f"True Positive (TP): {total_TP} / TP 平均 IoU: {avg_iou:.4f}\n")
        f.write(f"False Positive (FP): {total_FP}\n")
        f.write(f"False Negative (FN): {total_FN}\n")
        f.write(f"多餘框數量: {len(duplicate_list)}\n\n")

        f.write("## 錯誤影像快速檢視\n")
        f.write(f"- 合 FP 的圖片: {FP_list}\n")
        f.write(f"- 合 FN 的圖片: {FN_list}\n")
        f.write(f"- 有多框問題: {duplicate_list}\n")
        f.write(f"- 僅有 TP 的圖片: {TP_list}\n\n")

        f.write("## 圖像輸出分類（請自行建立資料夾）\n")
        f.write("- false_positive: analysis_output/visualizations/false_positive\n")
        f.write("- false_negative: analysis_output/visualizations/false_negative\n")
        f.write("- duplicates:     analysis_output/visualizations/duplicates\n")
        f.write("- true_positive:  analysis_output/visualizations/true_positive\n")

    print(f"分析完成！詳細報告已輸出：{REPORT_FILE}")


if __name__ == "__main__":
    main()
