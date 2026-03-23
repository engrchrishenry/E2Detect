import glob
import os
import time
import argparse
import numpy as np
import torch
from PIL import Image
from PIL import ImageFont
from vizer.draw import draw_boxes
from yacs.config import CfgNode as CN
from models_SSD.SSD import SSDDetector_event

if not hasattr(ImageFont.FreeTypeFont, "getsize"):
    def getsize(self, text):
        bbox = self.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    ImageFont.FreeTypeFont.getsize = getsize


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, feats_dir, output_dir):    
    class_names = (
            "__background__",
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ) # background + 20 classes
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SSDDetector_event(cfg)
    model.load_state_dict(torch.load(ckpt, map_location=device).pop("model"))
    model = model.to(device)
    print('Loaded weights from {}'.format(ckpt))

    image_paths = glob.glob(os.path.join(images_dir, '**', '*.jpg'), recursive=True) \
             + glob.glob(os.path.join(images_dir, '**', '*.png'), recursive=True)

    os.makedirs(f'{output_dir}/ssd_detections_vis', exist_ok=True)
    os.makedirs(f'{output_dir}/ssd_feat', exist_ok=True)

    cpu_device = torch.device("cpu")
    model.eval()
    all_detections = []
    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)
        image_name, image_ext = os.path.splitext(image_name)
        # image_name = f"{image_path.split('/')[-2]}_{image_path.split('/')[-1]}"
        # image_name, image_ext = os.path.splitext(image_name)

        feat = np.load(f'{feats_dir}/{image_name}.npz')['patch']
        feat = torch.from_numpy(feat).unsqueeze(0)
        feat = torch.clamp(feat, min=0.0)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        load_time = time.time() - start

        start = time.time()
        result, backbone_feature = model(feat.to(device))
        
        backbone_feature = backbone_feature.squeeze(0).cpu().numpy()
        np.savez_compressed(f'{output_dir}/ssd_feat/{image_name}', backbone_feature=backbone_feature)
        
        result = result[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()

        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        boxes = np.clip(boxes, 0, [width, height, width, height])

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]

        all_detections.append({
            'image_id': image_name,
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'scores': torch.tensor(scores, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        })
        
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name+image_ext, meters))
        
        drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, "ssd_detections_vis", f"{image_name}.jpg"))# , quality=50, optimize=True)
    
    torch.save(all_detections, os.path.join(output_dir, 'detections_from_feats.pth'))


def main():
    parser = argparse.ArgumentParser(description="Get detections from E-FPN-generated SSD backbone features.")
    parser.add_argument("--config", type=str, required=True,
                        help="path to config file",)
    parser.add_argument("--images_dir", type=str, required=True,
                        help='Specify image dir to do prediction.')
    parser.add_argument("--feats_dir", type=str, required=True,
                        help='Specify backbone feature dir to do prediction.')
    parser.add_argument("--output_dir", type=str, required=True,
                        help='Specify output directory to save predictions.')
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained weights file for SSD.")
    parser.add_argument("--score_threshold", type=float, default=0.7) # 0.7

    args = parser.parse_args()
    print(args)

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.freeze()

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             feats_dir=args.feats_dir,
             output_dir=args.output_dir)


if __name__ == '__main__':
    main()
