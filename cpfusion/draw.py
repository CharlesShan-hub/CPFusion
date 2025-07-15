import click
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from cslib.utils.config import Options

# Paths - llvip
default_ir_dir = "/Volumes/Charles/data/vision/torchvision/llvip/infrared/test"
default_vis_dir = "/Volumes/Charles/data/vision/torchvision/llvip/visible/test"
default_fused_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused"
default_res_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused"
default_res_name = "image.png"

# Paths - tno
# default_ir_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/ir"
# default_vis_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/vis"
# default_fused_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/fused"
# default_res_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/fused"
default_res_name = "image.png"

# Fusion Images - Calculare for specified images
default_img_id = ('190001','190002','190003','260098','260101')
# defaulf_img_id = ('190001','190002')

# Fusion Images Detail Part Info -> (x,y,w)
default_img_pos = ((700,480,60),(970,380,60),(950,0,60),(700,480,60),(700,480,60))

# Fusion Algorithms - `fused_dir` is the parent dir of all algorithms
default_algorithms = ('cpfusion','datfuse','fpde','fusiongan','gtf','ifevip','piafusion','stdfusion','tardal')
# default_algorithms = ('cpfusion',)

@click.command()
@click.option('--ir_dir', default=default_ir_dir)
@click.option('--vis_dir', default=default_vis_dir)
@click.option('--fused_dir', default=default_fused_dir)
@click.option('--algorithms', default=default_algorithms, multiple=True, help='draw for multiple fusion algorithms')
@click.option('--img_id', default=default_img_id, multiple=True, help='draw for specified images')
@click.option('--img_pos', default=default_img_pos, multiple=True, help='draw for specified images')
@click.option('--corner', default='SE',help='NW | NE | SW | SE')
@click.option('--suffix', default="jpg")
@click.option('--window_size', default=0.4)
@click.option('--thickness', default=8)
@click.option('--res_dir', default=default_res_dir, help='Path to save image.')
@click.option('--res_name', default=default_res_name, help='Name of image file.')
def main(**kwargs):
    # Image Info
    opts = Options('Draw Details',kwargs)
    example_img = Image.open(Path(opts.ir_dir) / f"{opts.img_id[0]}.{opts.suffix}")
    window_size = (int(example_img.width*opts.window_size), int(example_img.height*opts.window_size))
    if opts.corner == 'NW':
        detail_x_offset, detail_y_offset = 0, 0
    elif opts.corner == 'NE':
        detail_x_offset, detail_y_offset = example_img.width - window_size[0], 0
    elif opts.corner == 'SW':
        detail_x_offset, detail_y_offset = 0, example_img.height - window_size[1]
    elif opts.corner == 'SE':
        detail_x_offset, detail_y_offset = example_img.width - window_size[0], example_img.height - window_size[1]
    else:
        raise ValueError('`corner` can only be NW, NE, SW, SE')

    # Load images
    images = []
    detail_images = []
    for img, pos in zip(opts.img_id, opts.img_pos):
        # Origin Images
        ir_img = Image.open(Path(opts.ir_dir) / f"{img}.{opts.suffix}").resize(example_img.size)
        vis_img = Image.open(Path(opts.vis_dir) / f"{img}.{opts.suffix}").resize(example_img.size)
        fused_imgs = [Image.open(Path(opts.fused_dir) / alg / f"{img}.{opts.suffix}").resize(example_img.size) for alg in opts.algorithms]
        
        # Detail Images
        ir_detail_img = ir_img.crop((pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[2] / example_img.width * example_img.height)).resize(window_size)
        vis_detail_img = vis_img.crop((pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[2] / example_img.width * example_img.height)).resize(window_size)
        fused_detail_imgs = [img.crop((pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[2] / example_img.width * example_img.height)).resize(window_size) for img in fused_imgs]
        
        # Draw Boxes
        for img in [ir_img, vis_img]+fused_imgs:
            draw_origin = ImageDraw.Draw(img)
            draw_origin.rectangle((pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[2] / example_img.width * example_img.height), outline='green',width=opts.thickness)
        for detail_img in [ir_detail_img, vis_detail_img] + fused_detail_imgs:
            draw_detail = ImageDraw.Draw(detail_img)
            draw_detail.rectangle(((0,0), detail_img.size), outline='red',width=opts.thickness)
            
        images.append([ir_img, vis_img] + fused_imgs)
        detail_images.append([ir_detail_img, vis_detail_img] + fused_detail_imgs)

    # Calculate total width and height
    total_width = (len(opts.algorithms)+2) * example_img.width
    total_height = len(opts.img_id) * example_img.height
    
    # Create a new image and paste each image into the new image
    new_img = Image.new('RGB', (total_width, total_height))
    y_offset = 0
    for row, detail_row in zip(images, detail_images):
        x_offset = 0
        for img, detail_img in zip(row, detail_row):
            new_img.paste(img, (x_offset, y_offset))
            new_img.paste(detail_img, (x_offset + detail_x_offset, y_offset + detail_y_offset))
            x_offset += img.width
        y_offset += img.height
    
    # Save and display the new image
    output_path = Path(opts.res_dir) / opts.res_name
    new_img.save(output_path,)
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()