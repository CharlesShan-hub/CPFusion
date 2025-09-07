import click
from tqdm import tqdm

from cslib.utils import get_device
from cslib.utils.config import Options
from cslib.datasets.fusion import GeneralFusion
from cslib.metrics.fusion.utils import Database

# Paths - llvip
# default_ir_dir = "/Volumes/Charles/data/vision/torchvision/llvip/infrared/test"
# default_vis_dir = "/Volumes/Charles/data/vision/torchvision/llvip/visible/test"
# default_fused_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused"
# default_db_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused"
# default_db_name = "metrics.db"

# Paths - tno
# default_ir_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/ir"
# default_vis_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/vis"
# default_fused_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/fused"
# default_db_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/fused"
# default_db_name = "metrics.db"

# Path - msrs
default_ir_dir = "/Volumes/Charles/data/vision/torchvision/msrs/test/ir"
default_vis_dir = "/Volumes/Charles/data/vision/torchvision/msrs/test/vi"
default_fused_dir = "/Volumes/Charles/data/vision/torchvision/msrs/test/fused"
default_db_dir = "/Volumes/Charles/data/vision/torchvision/msrs/test/fused"
default_db_name = "metrics.db"
default_suffix = "png"

# Fusion Images
# 1. Calculare all images in each fused_dir
defaulf_img_id = ()
# 2. Calculare for specified images
# defaulf_img_id = ('190001','190002','190003','190004','190005')
# defaulf_img_id = ('39',)

# Fusion Algorithms
# 1. `fused_dir` is into one algorithm
# default_algorithms = () 
# 2. `fused_dir` is the parent dir of all algorithms
# default_algorithms = ('cpfusion','datfuse','fpde','fusiongan','gtf','ifevip','piafusion','stdfusion','tardal')
default_algorithms = ('tardal',)

# Metrics
# default_metrics = ['pfe']
# 1. All Metrics
# default_metrics = [
#     'ce','en','te','mi','nmi','q_ncie','psnr','cc','scc','scd',
#     'ssim','ms_ssim','q_s','q','q_w','q_e','q_c','q_y','mb','mae',
#     'mse','rmse','nrmse','ergas','d','ag','mg','ei','pfe','sd','sf',
#     'q_abf','q_sf','eva','sam','asm','con','fmi','n_abf','pww',
#     'q_cv','q_cb','vif'
# ]
default_metrics = [
    'ag','ei','en','scd','sf','vif','q_cv'
]

# 2. VIFB
# default_metrics = [
#     'ce','en','mi','psnr','ssim','rmse','ag','ei','sf',
#     'q_abf','sd','q_cb','q_cv'
# ]
# 3. MEFB
# default_metrics = [
#     'ce','en','fmi','nmi','psnr','q_ncie','te','ag','ei',
#     'q_abf','sd','sf','q_c','q_w','q_y','q_cb','q_cv','vif'
# ]

@click.command()
@click.option('--ir_dir', default=default_ir_dir)
@click.option('--vis_dir', default=default_vis_dir)
@click.option('--fused_dir', default=default_fused_dir)
@click.option('--algorithms', default=default_algorithms, multiple=True, help='compute metrics for multiple fusion algorithms')
@click.option('--img_id', default=defaulf_img_id, multiple=True, help='compute metrics for specified images')
@click.option('--metrics', default=default_metrics, multiple=True)
@click.option('--suffix', default=default_suffix)
@click.option('--db_dir', default=default_db_dir, help='Path to save database file.')
@click.option('--db_name', default=default_db_name, help='Name of database file.')
@click.option('--device', default='auto', help='auto | cuda | mps | cpu')
@click.option('--jump', default=True, help='Jump Metrics that calculated before.')
def main(**kwargs):
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('Compute Metrics',kwargs)
    opts.presentParameters()
    dataset = GeneralFusion(
        ir_dir = opts.ir_dir,
        vis_dir = opts.vis_dir,
        fused_dir = opts.fused_dir,
        suffix = opts.suffix,
        algorithms = opts.algorithms,
        img_id = opts.img_id,
    )
    database = Database(
        db_dir = opts.db_dir, 
        db_name = opts.db_name,
        metrics = opts.metrics,
        algorithms = opts.algorithms,
        jump = opts.jump,
        mode = 'compute' # compute就是把 metrics 和 algroithms 构建好 index
    )
    for idx in tqdm(range(len(dataset)), desc="Computing Metrics", unit="image"):
        item: dict = dataset[idx]
        database.compute(
            ir = item["ir"].to(opts.device),
            vis = item["vis"].to(opts.device),
            fused = item["fused"].to(opts.device),
            algorithm = item["algorithm"],
            img_id = item["id"],
            logging = False,
            commit = False,
        )
        if (idx % 10 == 0) or (idx == len(dataset)-1):
            database.commit()
    
if __name__ == '__main__':
    main()