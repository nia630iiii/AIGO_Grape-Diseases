arser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
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

    return bboxes_scaled,probas
   
   
