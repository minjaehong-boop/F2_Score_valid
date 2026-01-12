import numpy as np
import pandas as pd


def bbox_iou(box1, box2, eps=1e-7):
    """Calculate the Intersection over Union (IoU) between bounding boxes.

    Args:
        box1, box2 (numpy): A numpy array of bounding boxes, with the last dimension being 4.

    Returns:
        (torch.Tensor): IoU
    """
    # transform from xywh to xyxy
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1, box2
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    # Intersection area
    inter = (inter_x2 - inter_x1).clip(0) * (inter_y2 - inter_y1).clip(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    return iou


def f2_with_iou(gt, pr, th=0.01):
    tp_iou = []
    tp = []
    fp = []
    fn = []

    #  GT와 PR에 있는 모든 이미지 목록 가져오기
    all_images = set(gt["file_name"]) | set(pr["file_name"])

    for img in list(all_images):
        gt_img = gt[gt["file_name"] == img][["cx", "cy", "width", "height"]]
        pr_img = pr[pr["file_name"] == img][["cx", "cy", "width", "height"]]

        num_gt = len(gt_img)
        num_pr = len(pr_img)

        max_vals = np.array([])

        if num_pr > 0:  # 예측이 있는 경우
            if num_gt > 0:  # 정답도 있는 경우 (TP, FP, FN 계산)
                ious = [bbox_iou(i, j) for i in gt_img.values for j in pr_img.values]
                ioumat = np.array(ious).reshape(num_gt, -1)  # gt_dim:0, pr_dim:1
                print(f"ious matrix for image {img}: {ioumat}")

                # pr을 iou가 최대인 gt에 할당
                ioumat = ioumat * (ioumat.max(axis=0, keepdims=True) == ioumat)
                print(f"ioumat after assignment for image {img}: {ioumat}")

                # TP_IoU / FP / FN
                max_vals = np.amax(ioumat, axis=1)  # 각 GT별 최대 IoU

                matched_gt_mask = max_vals > th

                tp_iou.extend([i for i in max_vals[matched_gt_mask]])
                tp.append(np.sum(matched_gt_mask))  # th보다 큰 GT 수
                fn.append(num_gt - np.sum(matched_gt_mask))  # th보다 작거나 같은 GT 수

                # FP (중복 검출)
                fp.extend([sum(i > th) - 1 for i in ioumat if sum(i > th) >= 2])

                pr_max_ious = np.amax(ioumat, axis=0)  # 각 PR별 최대 IoU
                fp.append(np.sum(pr_max_ious <= th))  # th 이하로 매칭된 PR 수

                print(f"tp_iou: {tp_iou}, tp: {tp}, fp: {fp}, fn: {fn}")

            else:  # 정답은 없고 예측만 있는 경우 (모두 FP)
                fn.append(0)
                tp.append(0)
                fp.append(num_pr)  # 모든 예측이 FP
                print(f"Image {img}: No GT, all {num_pr} PRs are FP.")

        else:  # 예측이 없는 경우
            tp.append(0)
            fp.append(0)
            if num_gt > 0:  # 정답만 있는 경우
                fn.append(num_gt)
                print(f"Image {img}: No PR, all {num_gt} GTs are FN.")
            else:  # 정답도 예측도 없는 경우
                fn.append(0)
                print(f"Image {img}: No GT or PR.")

        print(f"--- Results for image {img} ---")
        print("IoU per GT: ", max_vals)
        print("TP (count): ", tp[-1] if len(tp) else 0)
        print("FP (count): ", fp[-1] if len(fp) else 0)
        print("FN (count): ", fn[-1] if len(fn) else 0)
        print("-" * (24 + len(img)))

    tp_iou = np.sum(tp_iou)
    tp = np.sum(tp)
    fp = np.sum(fp)
    fn = np.sum(fn)

    # precision
    precision_denominator = tp + fp
    precision = 0 if precision_denominator == 0 else tp_iou / precision_denominator
    # recall
    recall_denominator = tp + fn
    recall = 0 if recall_denominator == 0 else tp_iou / recall_denominator

    # F2 score
    beta_sq = 2**2
    f2_denominator = (beta_sq * precision) + recall
    f2_score = (
        0
        if f2_denominator == 0
        else (1 + beta_sq) * (precision * recall) / f2_denominator
    )

    print("\n--- Final Score ---")
    print(f"Total TP IoU Sum: {tp_iou}")
    print(f"Total TPs (count): {tp}")
    print(f"Total FPs (count): {fp}")
    print(f"Total FNs (count): {fn}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    return f2_score


gt = pd.read_csv("gt.csv")
pr = pd.read_csv("pr.csv")

print("F2-Score: ", f2_with_iou(gt, pr))
