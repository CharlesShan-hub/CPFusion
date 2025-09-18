from cslib.utils import *
import click
import numpy as np

@click.command()
@click.option('--img1_dir', default=r"/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/attension/8.png", help='Before PAM.')
@click.option('--img2_dir', default=r"/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/attension/12.png", help='After PAM.')
@click.option('--margin', default=10)
def draw(img1_dir, img2_dir, margin):
    if img1_dir is None or img2_dir is None:
        click.echo('Please input img1_dir and img2_dir.')
        return
    img1 = to_numpy(path_to_gray(img1_dir))
    img2 = to_numpy(path_to_gray(img2_dir))
    img3 = img1 * 1.4
    img = np.ones(shape=(img1.shape[0], img1.shape[1]*3 + margin*2))
    img[:, :img1.shape[1]] = img1
    img[:, img1.shape[1]+margin:img1.shape[1]*2+margin] = img2
    img[:, img1.shape[1]*2+2*margin:] = img3
    # glance([img])
    save_array_to_img(img, r"/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/attension/contract.png")


if __name__ == '__main__':
    draw()