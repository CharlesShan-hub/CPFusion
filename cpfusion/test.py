import click
from cslib.utils import glance, path_to_gray, path_to_rgb, save_array_to_img, to_tensor
from cslib.dataset.fusion import TNO, LLVIP
from torch.utils.data import DataLoader
from utils import *
from model import *
from inference import fusion
from pathlib import Path

@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--save_path", type=click.Path(exists=True), required=True)
@click.option("--layer", type=int, default=4)
def main(**kwargs):
    # dataset = LLVIP(root=kwargs['dataset_path'], transform=Compose([ToTensor()]), download=True, train=False)
    # dataset = LLVIP(root=kwargs['dataset_path'], transform=None, download=True, train=False)
    dataset = TNO(root=kwargs['dataset_path'], transform=Compose([ToTensor()]), download=True)
    # dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for i in range(len(dataset)):
        # if i%50 != 0:
        #     continue
        ir,vis = dataset[i]
        fused = Path(kwargs['save_path']) / Path(ir).name
        ir_img = to_tensor(path_to_gray(ir)).unsqueeze(0)
        vis_img = to_tensor(path_to_rgb(vis)).unsqueeze(0)
        fused_img = fusion(ir_img, vis_img, kwargs['layer'])
        # glance([ir,vis,fused])
        save_array_to_img(fused_img, filename=fused)

if __name__ == '__main__':
    main()
