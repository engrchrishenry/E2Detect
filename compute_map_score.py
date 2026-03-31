import torch
import numpy as np
import argparse
from eval_detection_voc import eval_detection_voc


def filter_class(dets, class_ids):

    class_ids = set(class_ids)
    filtered = []

    for d in dets:
        labels = d['labels']
        mask = torch.tensor([int(l.item()) in class_ids for l in labels], dtype=torch.bool, device=labels.device)
        filtered.append({
            'boxes':  d['boxes'][mask],
            'labels': d['labels'][mask],
            'scores': d['scores'][mask]
        })

    return filtered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script computes VOC-style mAP.')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to the baseline detections .pth file')
    parser.add_argument('--e_fpn', type=str, required=True,
                        help='Path to the E-FPN detections .pth file')
    parser.add_argument('--selected_ids', type=int, nargs='+', required=True,
                        help='List of selected class ids to evaluate. E.g., 15 for "person" class in VOC')
    args = parser.parse_args()
    
    # ---- Load detections ----
    baseline = torch.load(args.baseline)   # RGB-based SSD detections
    e_fpn   = torch.load(args.e_fpn)   # E-FPN feature-based detections
    selected_ids = [15]
    
    class_names = ('__background__',
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor')

    baseline = filter_class(baseline, selected_ids)
    e_fpn   = filter_class(e_fpn, selected_ids)

    ious = np.arange(0.5, 1.0, 0.05)  # 0.50 → 0.95
    aps = []

    for iou in ious:
        res = eval_detection_voc(
            pred_bboxes=[d['boxes'].cpu().numpy() for d in e_fpn],
            pred_labels=[d['labels'].cpu().numpy() for d in e_fpn],
            pred_scores=[d['scores'].cpu().numpy() for d in e_fpn],
            gt_bboxes=[d['boxes'].cpu().numpy() for d in baseline],
            gt_labels=[d['labels'].cpu().numpy() for d in baseline],
            iou_thresh=iou,
            use_07_metric=False
        )
        aps.append(np.nanmean(res['ap']))  # mAP for this IoU

    aps = np.array(aps)
    map_05 = aps[0]           # AP@0.5
    map_075 = aps[5]          # AP@0.75
    map_05_095 = aps.mean()   # mAP@[0.5:0.95]

    print(f"AP@0.5 = {map_05:.3f}")
    print(f"AP@0.75 = {map_075:.3f}")
    print(f"mAP@[0.5:0.95] = {map_05_095:.3f}")
    
