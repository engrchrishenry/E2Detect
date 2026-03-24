import torch
import numpy as np
from eval_detection_voc import eval_detection_voc


def filter_class(dets, class_ids):
    # if isinstance(class_ids, int):
    #     class_ids = [class_ids]  # convert single ID to list

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
    # ---- Load detections ----
    baseline = torch.load("/storage4tb/PycharmProjects/rpg_e2vid/output/updated/esim_reds_filtered_patched/val/5_0.55_0.005_50_70000_300000/detections_from_images.pth")   # RGB-based SSD detections
    tsfnet   = torch.load("output/det_cos_normed/detections_from_feats.pth")   # TSFNet feature-based detections
    # tsfnet   = torch.load("/storage4tb/PycharmProjects/rpg_e2vid/output/updated/esim_reds_filtered_patched/val/5_0.55_0.005_50_70000_300000/detections_from_images.pth")   # RGB-based SSD detections
    out_dir = 'demo/'
    out_f_name = 'map_cos_normed.txt'
    use_07_metric = True # True False
    selected_ids = [15] # [15] # [15, 4, 7]
    
    class_names = ('__background__',
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor')

    baseline = filter_class(baseline, selected_ids)
    tsfnet   = filter_class(tsfnet, selected_ids)

    ious = np.arange(0.5, 1.0, 0.05)  # 0.50 → 0.95
    aps = []

    for iou in ious:
        res = eval_detection_voc(
            pred_bboxes=[d['boxes'].cpu().numpy() for d in tsfnet],
            pred_labels=[d['labels'].cpu().numpy() for d in tsfnet],
            pred_scores=[d['scores'].cpu().numpy() for d in tsfnet],
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
    exit(0)


    # shared_classes = sorted(set(
    #     torch.cat([d['labels'] for d in baseline]).unique().tolist()
    # ).intersection(
    #     torch.cat([d['labels'] for d in tsfnet]).unique().tolist()
    # ))

    baseline_unique_labels, baseline_counts = np.unique(np.concatenate([d['labels'].cpu().numpy() for d in baseline]), return_counts=True)
    tsfnet_unique_labels, tsfnet_counts = np.unique(np.concatenate([d['labels'].cpu().numpy() for d in tsfnet]), return_counts=True)
    common_labels = np.intersect1d(baseline_unique_labels, tsfnet_unique_labels)
    baseline_unique_label_names = [[class_names[d], c] for d, c in zip(baseline_unique_labels, baseline_counts)]
    tsfnet_unique_label_names = [[class_names[d], c] for d, c in zip(tsfnet_unique_labels, tsfnet_counts)]

    print(f'baseline_unique_labels = {baseline_unique_labels}')
    print(f'baseline_unique_label_names = {baseline_unique_label_names}')
    print(f'tsfnet_unique_labels = {tsfnet_unique_labels}')
    print(f'tsfnet_unique_label_names = {tsfnet_unique_label_names}')
    print(f'common_labels = {common_labels}')

    pred_boxes_list  = [d['boxes'].cpu().numpy()  for d in tsfnet]
    pred_labels_list = [d['labels'].cpu().numpy()  for d in tsfnet]
    pred_scores_list = [d['scores'].cpu().numpy()  for d in tsfnet]
    gt_boxes_list     = [d['boxes'].cpu().numpy()    for d in baseline]
    gt_labels_list    = [d['labels'].cpu().numpy()   for d in baseline]

    assert len(pred_boxes_list) == len(gt_boxes_list)

    # result = eval_detection_voc(pred_bboxes=pred_boxes_list,
    #                                 pred_labels=pred_labels_list,
    #                                 pred_scores=pred_scores_list,
    #                                 gt_bboxes=gt_boxes_list,
    #                                 gt_labels=gt_labels_list,
    #                                 # gt_difficults=gt_difficults,
    #                                 iou_thresh=0.5,
    #                                 use_07_metric=use_07_metric)

    # Evaluate baseline (SSD)
    res_ssd = eval_detection_voc(
        pred_bboxes=[d['boxes'].cpu().numpy() for d in baseline],
        pred_labels=[d['labels'].cpu().numpy() for d in baseline],
        pred_scores=[d['scores'].cpu().numpy() for d in baseline],
        gt_bboxes=[d['boxes'].cpu().numpy() for d in baseline],
        gt_labels=[d['labels'].cpu().numpy() for d in baseline],
        iou_thresh=0.5,
        use_07_metric=use_07_metric
    )

    # Evaluate TSFNet
    res_tsf = eval_detection_voc(
        pred_bboxes=[d['boxes'].cpu().numpy() for d in tsfnet],
        pred_labels=[d['labels'].cpu().numpy() for d in tsfnet],
        pred_scores=[d['scores'].cpu().numpy() for d in tsfnet],
        gt_bboxes=[d['boxes'].cpu().numpy() for d in baseline],
        gt_labels=[d['labels'].cpu().numpy() for d in baseline],
        iou_thresh=0.5,
        use_07_metric=use_07_metric
    )

    print (res_tsf)
    for id in selected_ids:
        print(f"{class_names[id]} AP:", res_tsf["ap"][id])
    print (f'mAP: {res_tsf["map"]}')

    prec_ssd, rec_ssd = res_ssd['prec'][selected_ids[0]], res_ssd['rec'][selected_ids[0]]
    prec_tsf, rec_tsf = res_tsf['prec'][selected_ids[0]], res_tsf['rec'][selected_ids[0]]

    if prec_ssd is None or rec_ssd is None or prec_tsf is None or rec_tsf is None:
        raise ValueError("Person class not found in one of the detectors!")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(rec_tsf, prec_tsf, linewidth=2, label="TSFNet (ours)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Person Class)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_curve_person_tsfnet.png", dpi=300)


    exit(0)
    
    result_str = "mAP: {:.4f}\n".format(result["map"])
    metrics = {'mAP': result["map"]}
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        metrics[class_names[i]] = ap
        result_str += "{:<16}: {:.4f}\n".format(class_names[i], ap)
    print (result_str)

    with open(f'{out_dir}/{out_f_name}', "w") as f:
        f.write(result_str)
