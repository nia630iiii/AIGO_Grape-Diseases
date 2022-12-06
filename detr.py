import os
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from models import build_model
from datasets.grape import make_grape_transforms

def get_args_parser():#參數輸入
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='grape')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='./checkpoint.pth', help='resume from checkpoint')

    parser.add_argument('--threshold', default=0.5, type=float)

    return parser

def box_cxcywh_to_xyxy(x):#bbox格式轉換
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):#rescale bbox
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def crop_one_grape(img, bboxes_scaled, output_path):#裁剪葡萄圖片
    n = 0
    for i in bboxes_scaled:
        if(i[0] < 0):
            i[0] = 0
        if(i[1] < 0):
            i[1] = 0
        if(i[2] < 0):
            i[2] = 0
        if(i[3] < 0):
            i[3] = 0
        im = img[int(i[1]):int(i[3]),int(i[0]):int(i[2])]
        if not os.path.exists(output_path + "single"):
            os.mkdir(output_path + "single")
        save_path_file = os.path.join(output_path + "single" + "/" +str(n)+".jpg")
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path_file, im)
        n += 1

@torch.no_grad()
def infer(img_sample, model, device, threshold, output_path):
    model.eval()
    filename = os.path.basename(img_sample)
    orig_image = Image.open(img_sample)
    orig_image = ImageOps.exif_transpose(orig_image)
    w, h = orig_image.size
    transform = make_grape_transforms("val")
    dummy_target = {
        "size": torch.as_tensor([int(h), int(w)]),
        "orig_size": torch.as_tensor([int(h), int(w)])
    }
    image, targets = transform(orig_image, dummy_target)
    image = image.unsqueeze(0)
    
    image = image.to(device)

    outputs = model(image)
    outputs["pred_logits"] = outputs["pred_logits"].cpu()
    outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], orig_image.size)
    probas = probas[keep].cpu().data.numpy()

    if len(bboxes_scaled) == 0:
        pass
    img = np.array(orig_image)

    img_save_path = os.path.join(output_path, filename)
    crop_one_grape(img, bboxes_scaled, output_path)
    single_normal, single_bitter, single_ripe = plot_results(img, img_save_path, probas, bboxes_scaled, threshold)
    print("正常葡萄數量:", single_normal)
    print("苦腐病葡萄數量:", single_bitter)
    print("晚腐病葡萄數量:", single_ripe)

def plot_results(pil_img, prob, boxes, thresh):
    COLORS = ["#00ff00", "#8B0000", "#E60000", "#B8860B", "#FFD700"]
    CLASSES = ['normal', 'earlybitter', 'bitter', 'earlyripe', 'ripe']
    bitter = 0
    ripe = 0
    normal = 0
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        if CLASSES[cl]=="normal":
            thick = 2
            normal += 1
        elif CLASSES[cl]=="earlybitter" or CLASSES[cl]=="bitter":
            thick = 2
            bitter += 1
        elif CLASSES[cl]=="earlyripe" or CLASSES[cl]=="ripe":
            thick = 2
            ripe += 1
        else:
            thick = 2.5
        if(colors[cl]=="#00ff00"):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color=colors[cl], linewidth=thick))
        else:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=True, color=colors[cl], linewidth=thick,alpha = 0.5))
                                
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=8,
                bbox=dict(facecolor='white', alpha = thresh))
    plt.axis('off')
    plt.show()
    return normal, bitter, ripe

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    output_dir = './result/'

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)

    image_paths = "./disease_0007.jpg"

    infer(image_paths, model, device, args.threshold, output_dir)