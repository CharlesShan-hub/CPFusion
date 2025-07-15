'''
Transform to pyramid by adaptive Gaussian kernel
'''

import click
from cslib.utils import glance
from cslib.datasets.fusion import LLVIP, TNO
from cslib.algorithms.msd import Laplacian, Graident
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
def main(**kwargs):
    dataset = LLVIP(root=kwargs['dataset_path'], transform=Compose([ToTensor()]), download=True, train=False)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    for ir,vis in dataloader:
        pyr = Laplacian(image = ir)
        if isinstance(pyr, Laplacian):
            layers = pyr.pyramid
            recon = pyr.recon
            glance([ir,recon])
            glance(layers)
        elif isinstance(pyr, Graident):
            layers = pyr.pyramid
            recon = pyr.recon
            glance([ir,recon])
        break


if __name__ == '__main__':
    main()