import click
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from cslib.utils.config import Options

import matplotlib
# other backends:
# https://matplotlib.org/stable/users/explain/figure/backends.html
matplotlib.use('macosx')

cc1_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused/cc1"
max1_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused/max1"
cc_max1_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused/cc_max1"
cc2_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused/cc2"
max2_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused/max2"
cc_max2_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused/cc_max2"

@click.command()
@click.option('--cc1', default=cc1_dir)
@click.option('--max1', default=max1_dir)
@click.option('--cc_max1', default=cc_max1_dir)
@click.option('--cc2', default=cc2_dir)
@click.option('--max2', default=max2_dir)
@click.option('--cc_max2', default=cc_max2_dir)
def main(**kwargs):
    opts = Options('Draw Details',kwargs)


if __name__ == '__main__':
    main()