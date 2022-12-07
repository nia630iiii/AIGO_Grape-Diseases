import torch
from torchvision.models import resnet18
import random


def padim(data_path,checkpoint_pth):

    # device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # load model
    model = resnet18(pretrained=False, progress=True)
    t_d = 448
    d = 100

    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
   