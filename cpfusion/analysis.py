import click
import json
from cslib.utils.config import Options
from cslib.metrics.fusion.utils import Database

# Paths - m3fd
# default_db_dir = "/Users/kimshan/Public/data/vision/torchvision/m3fd/fused"
# default_db_name = "metrics.db"

# Paths - llvip
# default_db_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused"
# default_db_name = "metrics.db"

# default_db_dir = "/Volumes/Charles/data/vision/torchvision/llvip/fused/by_metrics"
# default_db_name = "ce.db"

# Paths - tno
default_db_dir = "/Volumes/Charles/data/vision/torchvision/tno/tno/fused"
default_db_name = "metrics.db"

# Fusion Images
# 1. Calculare all images in each fused_dir
default_img_id = ()
# Selected TNO
# default_img_id = ('45', '28', '5', '26', '58', '7', '53', '25', '48', '6', '30', '36', '1', '23', '21', '54', '12', '22', '41', '32', '34', '44', '3', '2', '59', '50', '11', '33', '209', '8', '46', '219', '199', '192', '18', '212', '213', '35', '193', '42', '195', '211', '210', '217', '216', '191', '49', '221', '52', '200', '194', '214', '201', '215', '206', '218', '20', '14', '196', '220', '205', '190', '10', '198', '203', '13', '17', '55', '19', '39', '202', '37', '4', '237', '239', '236', '234', '38', '232', '233', '9', '235', '238', '197', '229', '230', '231', '224', '228', '223', '226', '227', '225', '40', '16', '43', '88', '77', '68', '69')
# Selected LLVIP
# default_img_id = ('190015', '220305', '220186', '220098', '220190', '220101', '220004', '220108', '220316', '220300', '220112', '220028', '220073', '220055', '220376', '220176', '220277', '220228', '220093', '220045', '220015', '220246', '220027', '220359', '220268', '220154', '220188', '220053', '220380', '220122', '220377', '260270', '220158', '220023', '220220', '220230', '220205', '260536', '220125', '220171', '260359', '260253', '260065', '260382', '260321', '260377', '220075', '260239', '220308', '260241', '260520', '260460', '260306', '260028', '260508', '260358', '220166', '260484', '260248', '260493', '260468', '260528', '260118', '220285', '260374', '260071', '260171', '260127', '260254', '260311', '260307', '260344', '260339', '260406', '260123', '260134', '260185', '260209', '260107', '260150', '260525', '260457', '260243', '260089', '260222', '260504', '260476', '260281', '260011', '260423', '260006', '260084', '260507', '260442', '260392', '260305', '260015', '260198', '260290', '260001', '260456', '260337', '260360', '260168', '260043', '260091', '220374', '260188', '260034', '260156', '260274', '260378', '260490', '260532', '260200', '220131', '260247', '260437', '260061', '220078', '260215', '260080', '260201', '260236', '260221', '260269', '260412', '260055', '260471', '220162', '260308', '260432', '260410', '260357', '260147', '260172', '260007', '260346', '260486', '220178', '260070', '260265', '260224', '260251', '220034', '220206', '260090', '260196', '260096', '260262', '260031', '260010', '260288', '260086', '260294', '260519', '220067', '220012', '260175', '220126', '260256', '260416', '260293', '260369', '260052', '260114', '260411', '260068', '260109', '260186', '260487', '260438', '260013', '260354', '260515', '260252', '260482', '260126', '260325', '260295', '260079', '260440', '260475', '260448', '220130', '260016', '260203', '220151', '260417', '210326', '260396', '260483', '260489', '220256', '260390', '220379', '260461', '260088', '260407', '260474')

# # 2. Calculare for specified images
# default_img_id = ('190001','190002','190003')
# default_img_id = ('39',)

# Fusion Algorithms
# 1. `fused_dir` is into one algorithm
# default_algorithms = () 
# 2. `fused_dir` is the parent dir of all algorithms
# default_algorithms = ('SceneFuse','GTF','SDCFusion','DATFuse','VSMWLS','HMSD')
# default_algorithms = ('cpfusion','datfuse','fpde','fusiongan','gtf','ifevip','piafusion','stdfusion','tardal')
default_algorithms = ('cpfusion','datfuse','fpde','fusiongan','gtf','ifevip','piafusion','stdfusion','tardal')
# default_algorithms = ('cpfusion','cpfusion_wp','cpfusion_cc','cpfusion_max','datfuse','fpde','fusiongan','gtf','ifevip','piafusion','stdfusion','tardal')
# default_algorithms = ('cpfusion','cpfusion_max','cpfusion_cc')

# Metrics
default_metrics = [
    'ag','ei','en','scd','sf','vif','q_cv'
]
# default_metrics = [
#     'ce','en','te','mi','nmi','q_ncie','psnr','cc','scc','scd',
#     'ssim','ms_ssim','q_s','q','q_w','q_e','q_c','q_y','mb','mae',
#     'mse','rmse','nrmse','ergas','d','ag','mg','ei','pfe','sd','sf',
#     'q_abf','q_sf','eva','sam','asm','con','fmi','n_abf','pww',
#     'q_cv','vif'#, 'q_cb'
# ]
# big_metrics = [
#     'ag','ei','en','q_abf','q_cb','sf','vif'
# ]
# big_metrics = [
#     'ag','ei','en','q_abf','q_cb','sf','vif'
# ]
# small_metrics = [
#     'q_cv',
# ]
# small_metrics = [
#     'q_cv',
# ]
# default_metrics = big_metrics + small_metrics
# 1. All Metrics
# default_metrics = [
#     'ce','en','te','mi','nmi','q_ncie','psnr','cc','scc','scd',
#     'ssim','ms_ssim','q_s','q','q_w','q_e','q_c','q_y','mb','mae',
#     'mse','rmse','nrmse','ergas','d','ag','mg','ei','pfe','sd','sf',
#     'q_abf','q_sf','eva','sam','asm','con','fmi','n_abf','pww',
#     'q_cv','q_cb','vif'
# ]
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
@click.option('--metrics', default=default_metrics, multiple=True)
@click.option('--algorithms', default=default_algorithms, multiple=True, help='analyze metrics for multiple fusion algorithms')
@click.option('--db_dir', default=default_db_dir, help='Path to save database file.')
@click.option('--db_name', default=default_db_name, help='Name of database file.')
def main(**kwargs):
    opts = Options('Analyze Metrics',kwargs).parse({},present=True)
    database = Database(
        db_dir = opts.db_dir, 
        db_name = opts.db_name,
        metrics = opts.metrics,
        algorithms = opts.algorithms,
        mode = 'analyze' # analyze 就是检查 metrics 和 algorithms 已经存在
    )
    # print(json.dumps(database.analyze_average(), indent=4, sort_keys=True))
    print(json.dumps(database.analyze_average(img_id=default_img_id), indent=4, sort_keys=True))
    # print(json.dumps(database.analyze_general(), indent=4, sort_keys=False))
if __name__ == '__main__':
    main()
