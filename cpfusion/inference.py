import click
from typing import Union
import torch
from cslib.utils.image import to_tensor, rgb_to_ycbcr, gray_to_rgb, ycbcr_to_rgb, save_array_to_img, path_to_gray, path_to_rgb
from cslib.utils import get_device, Options, glance
from cslib.algorithms.msd import Laplacian, Contrust
from cslib.models import SimAMBlock
from utils import *
from model import *
import copy
from pathlib import Path


__all__ = [
    'fusion'
]

def _c(image: torch.Tensor) -> torch.Tensor:
    ''' 将灰度图转换为RGB图, 为了方便可视化
    '''
    B,_,H,W = image.shape
    res = torch.zeros(size=(B,3,H,W))
    res[:,0:1,:,:] = image
    res[:,1:3,:,:] = 128.0
    return to_tensor(ycbcr_to_rgb(res))

def image_init(ir: torch.Tensor, vis: torch.Tensor, debug: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert ir.shape[1] == 1 and ir.ndim == 4 and vis.ndim == 4
    if vis.shape[1] == 1:
        vis = to_tensor(gray_to_rgb(vis))
    vis_ycbcr = to_tensor(rgb_to_ycbcr(vis))
    vis_y = vis_ycbcr[:, :1, :, :]
    ir_y = to_tensor(rgb_to_ycbcr(gray_to_rgb(ir)))[:, :1, :, :]
    if debug:
        glance(
            [vis, _c(vis_y), _c(ir_y)],
            title=['VIS', 'VIS Y', 'IR Y'],
            shape=(1,3), 
            suptitle = 'Image Initialization',
            each_save=True,
            each_save_dir='./glance_outputs/origin'
        )
    return vis_ycbcr, vis_y, ir_y

def apple_msd(ir_y: torch.Tensor, vis_y: torch.Tensor, layer: int, msd_method: str, debug: bool) -> tuple[Union[Laplacian, Contrust], Union[Laplacian, Contrust]]:
    if msd_method == 'Laplacian':
        ir_pyr = Laplacian(image = ir_y, layer = layer, gau_blur_way = 'Adaptive')
        vis_pyr = Laplacian(image = vis_y, layer = layer, gau_blur_way = 'Adaptive')
    elif msd_method == 'Contrust':
        ir_pyr = Contrust(image = ir_y, layer = layer, gau_blur_way = 'Adaptive')
        vis_pyr = Contrust(image = vis_y, layer = layer, gau_blur_way = 'Adaptive')
    else:
        raise ValueError(f'Unknown msd method: {msd_method}')
    if debug:
        glance(
            [_c(ir_pyr.recon), _c(vis_pyr.recon), _c(ir_y), _c(vis_y)], 
            title=['IR rebuild', 'VIS rebuild', 'IR', 'VIS'],
            shape=(2,2), 
            suptitle = 'Multi-Scale Decomposition (result)',
            each_save=True,
            each_save_dir='./glance_outputs/msd'
            )
    return ir_pyr, vis_pyr

def get_base(ir_pyr: Union[Laplacian, Contrust], vis_pyr: Union[Laplacian, Contrust], layer: int, debug: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ir_base = ir_pyr.gaussian
    vis_base = vis_pyr.gaussian
    if debug:
        glance(
            [torch.abs(_c(i)) for i in ir_base] + [torch.abs(_c(i)) for i in vis_base],
            title=[f'Ir Base L{i+1}' for i in range(layer+1)] + [f'Vis Base L{i+1}' for i in range(layer+1)],
            shape=(2,layer+1),
            suptitle = 'Multi-Scale Decomposition (Base)',
            each_save=True,
            each_save_dir='./glance_outputs/base'
        )
    ir_base = torch.cat(msd_align(ir_base),dim=1)
    vis_base = torch.cat(msd_align(vis_base),dim=1)
    return ir_base, vis_base

def get_detail(ir_pyr: Union[Laplacian, Contrust], vis_pyr: Union[Laplacian, Contrust], layer: int, debug: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ir_detail = ir_pyr.pyramid
    vis_detail = vis_pyr.pyramid
    if debug:
        glance(
            [torch.abs(_c(i)) for i in ir_detail] + [torch.abs(_c(i)) for i in vis_detail],
            title=[f'Ir Detail L{i+1}' for i in range(layer)] + [f'Vis Detail L{i+1}' for i in range(layer)],
            shape=(2,layer),
            suptitle = 'Multi-Scale Decomposition (Detail)',
            each_save=True,
            each_save_dir='./glance_outputs/detail'
        )
    ir_detail = torch.cat(msd_align(ir_detail),dim=1)
    vis_detail = torch.cat(msd_align(vis_detail),dim=1)
    return ir_detail, vis_detail

def base_layer_fuse(ir_base: torch.Tensor, vis_base: torch.Tensor, fusion_method: str, layer: int, debug: bool) -> torch.Tensor:
    if fusion_method == 'CC+MAX':
        wcc = correlation_coefficient_weights(ir_base, vis_base)
        fused_base = _base_layer_fuse(ir_base, vis_base, wcc)
        if debug:
            # glance(
            #     [torch.abs(_c(wcc[:,i:i+1,:,:])) for i in range(layer)],
            #     title=[f'f_b{i+1} wcc={wcc[0,i:i+1,0,0].item()}' for i in range(layer)],
            #     suptitle = 'wcc',
            #     each_save=True,
            #     each_save_dir='./glance_outputs/wcc'
            # )
            glance(
                [torch.abs(_c(fused_base[:,i:i+1,:,:])) for i in range(layer)],
                title=[f'f_b{i+1} wcc={wcc[0,i:i+1,0,0].item()}' for i in range(layer)],
                suptitle = 'CC+MAX correlation coefficient weights',
                each_save=True,
                each_save_dir='./glance_outputs/cc_max'
            )
    elif fusion_method == 'CC':
        wcc = correlation_coefficient_weights(ir_base, vis_base)
        fused_base = ir_base * wcc + vis_base * (1 - wcc)
        if debug:
            glance(
                [torch.abs(_c(fused_base[:,i:i+1,:,:])) for i in range(layer)],
                title=[f'f_b{i+1} wcc={wcc[0,i:i+1,0,0].item()}' for i in range(layer)],
                suptitle = 'CC correlation coefficient weights',
                each_save=True,
                each_save_dir='./glance_outputs/cc'
            )
    elif fusion_method == 'MAX':
        fused_base = torch.max(ir_base, vis_base)
        if debug:
            glance(
                [torch.abs(_c(fused_base[:,i:i+1,:,:])) for i in range(layer)],
                title=[f'f_b{i+1}' for i in range(layer)],
                suptitle = 'max',
                each_save=True,
                each_save_dir='./glance_outputs/max'
            )
    else:
        raise ValueError(f'Unknown fusion method: {fusion_method}')
    return fused_base

def detail_layer_fuse(ir_detail: torch.Tensor, vis_detail: torch.Tensor, attension: str, layer: int, debug: bool) -> torch.Tensor:
    if attension != "None":
        if attension == 'SimAM':
            attension_block = SimAMBlock()  
        elif attension == 'DSimAM':
            attension_block = DSimAMBlock()
        else:
            raise ValueError(f'Unknown attension method: {attension}')
        ir_detail_enhanced = attension_block(ir_detail)
        vis_detail_enhanced = attension_block(vis_detail)
        if debug:
            glance(
                [torch.abs(_c(ir_detail[:,i:i+1,:,:])) for i in range(layer)]+\
                [torch.abs(_c(ir_detail_enhanced[:,i:i+1,:,:])) for i in range(layer)]+\
                [torch.abs(_c(vis_detail[:,i:i+1,:,:])) for i in range(layer)]+\
                [torch.abs(_c(vis_detail_enhanced[:,i:i+1,:,:])) for i in range(layer)],
                title=[f'irl{i+1}' for i in range(layer)]+\
                [f'ir with Sim l{i+1}' for i in range(layer)]+\
                [f'vis l{i+1}' for i in range(layer)]+\
                [f'vis with Sim l{i+1}' for i in range(layer)],
                shape=(4,layer), suptitle = 'Attension (Detail)', tight_layout=True,
                each_save=True,
                each_save_dir='./glance_outputs/attension'
            )
    else:
        ir_detail_enhanced = ir_detail
        vis_detail_enhanced = vis_detail

    # 细节层融合
    fused_detail = _detail_layer_fuse(ir_detail_enhanced, vis_detail_enhanced)
    if debug:
        glance(
            [torch.abs(_c(fused_detail[:,i:i+1,:,:])) for i in range(layer)],
            title=[f'f_detail_{i+1}' for i in range(layer)],
            shape=(1,layer), suptitle = 'Fusion (Detail)', tight_layout=True,
            each_save=True,
            each_save_dir='./glance_outputs/fusion_detail'
        )
    return fused_detail

def reconstruction(fused_base: torch.Tensor, fused_detail: torch.Tensor, ir_pyr: Union[Laplacian, Contrust], vis_ycbcr: torch.Tensor) -> torch.Tensor:
    fused_base = msd_resample(fused_base)
    fused_detail = msd_resample(fused_detail)
    fused_pyr = copy.deepcopy(ir_pyr)
    fused_pyr.gaussian = fused_base
    fused_pyr.pyramid = fused_detail
    fused_pyr.reconstruction()
    fused = copy.deepcopy(vis_ycbcr)
    fused[:,0:1,:,:] = fused_pyr.recon
    return to_tensor(ycbcr_to_rgb(fused)).clip(max=1.0, min=0.0)

def fusion(
        ir: torch.Tensor, 
        vis: torch.Tensor, 
        layer: int = 4, 
        debug: bool = False,
        msd_method: str = ['Laplacian','Contrust'][0],
        fusion_method: str = ['CC+MAX','CC','MAX'][0], 
        attension: str = ['SimAM','DSimAM','None'][1],
    ) -> torch.Tensor:
    # 得到Y通道
    vis_ycbcr, vis_y, ir_y = image_init(ir, vis, debug)

    # 多尺度分解
    ir_pyr, vis_pyr = apple_msd(ir_y, vis_y, layer, msd_method, debug)

    # 基础层+对齐
    ir_base, vis_base = get_base(ir_pyr, vis_pyr, layer, debug)

    # 细节层+对齐
    ir_detail, vis_detail = get_detail(ir_pyr, vis_pyr, layer, debug)

    # 细节层融合 - pam
    fused_detail = detail_layer_fuse(ir_detail, vis_detail, attension, layer, debug)

    # 基础层融合 - wcc
    fused_base = base_layer_fuse(ir_base, vis_base, fusion_method, layer, debug)

    # 重构图像 - 下采样 + 恢复成 RGB
    fused = reconstruction(fused_base, fused_detail, ir_pyr, vis_ycbcr)
    
    return fused

@click.command()
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--fusion_strategy", type=str, default=['CC+MAX','CC','MAX'][0])
@click.option("--pam_module", type=bool, default=True)
@click.option("--device", type=str, default='auto')
def main(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion', kwargs)
    opts.present()

    image_index = 190015
    ir = path_to_gray(f'/Volumes/Charles/data/vision/torchvision/llvip/infrared/test/{image_index}.jpg')
    vis = path_to_rgb(f'/Volumes/Charles/data/vision/torchvision/llvip/visible/test/{image_index}.jpg')

    # image_index = 48
    # ir = path_to_gray('/Users/kimshan/Public/project/paper/ir_250423.jpg')
    # vis = path_to_rgb('/Users/kimshan/Public/project/paper/vis_250423.jpg')
    # ir = path_to_gray('/Users/kimshan/Public/project/paper/ir_010379.jpg')
    # vis = path_to_rgb('/Users/kimshan/Public/project/paper/vis_010379.jpg')

    # image_index = 2
    # ir = path_to_gray(f'/Volumes/Charles/data/vision/torchvision/tno/tno/ir/{image_index}.png')
    # vis = path_to_rgb(f'/Volumes/Charles/data/vision/torchvision/tno/tno/vis/{image_index}.png')
    # ir = path_to_gray(f'/Users/kimshan/Public/data/vision/torchvision/tno/tno/lwir/{image_index}.png')
    # vis = path_to_gray(f'/Users/kimshan/Public/data/vision/torchvision/tno/tno/vis/{image_index}.png')
    
    ir = to_tensor(ir).unsqueeze(0).to(opts.device)
    vis = to_tensor(vis).unsqueeze(0).to(opts.device)

    glance([ir,vis,fusion(ir, vis, kwargs['layer'], debug=True)],title=['ir','vis','fused'],auto_contrast=False,clip=True,each_save=True,each_save_dir="./glance_outputs/final")
    # save_array_to_img(fusion(ir, vis, kwargs['layer'], debug=False), filename=f'/Volumes/Charles/data/vision/torchvision/tno/tno/fused/cpfusion/{image_index}.png')

@click.command()
@click.option("--p", type=str, default="/Volumes/Charles/data/vision/torchvision/tno/tno")
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--device", type=str, default='auto')
def test_tno(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion TNO', kwargs)
    opts.present()
    for i in (Path(opts.p) / 'ir').glob("*.png"):
        ir = to_tensor(path_to_gray(i)).to(opts.device)
        vis = path_to_rgb(Path(opts.p) / 'vis' / i.name)
        ir = to_tensor(ir).unsqueeze(0).to(opts.device)
        vis = to_tensor(vis).unsqueeze(0).to(opts.device)
        fused = fusion(ir, vis, kwargs['layer'], debug=False)
        name = Path(opts.p) / 'fused' / 'cpfusion' / i.name
        print(f"Saving {name}")
        save_array_to_img(fused, name, True)


@click.command()
@click.option("--p", type=str, default="/Volumes/Charles/data/vision/torchvision/tno/tno")
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--device", type=str, default='auto')
def test_tno_ablation_pam(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion without PAM TNO', kwargs)
    opts.present()
    for i in (Path(opts.p) / 'ir').glob("*.png"):
        ir = to_tensor(path_to_gray(i)).to(opts.device)
        vis = path_to_rgb(Path(opts.p) / 'vis' / i.name)
        ir = to_tensor(ir).unsqueeze(0).to(opts.device)
        vis = to_tensor(vis).unsqueeze(0).to(opts.device)
        fused = fusion(ir, vis, kwargs['layer'], debug=False, attension='None')
        name = Path(opts.p) / 'fused' / 'cpfusion_wp' / i.name
        print(f"Saving {name}")
        save_array_to_img(fused, name, True)



@click.command()
@click.option("--p", type=str, default="/Volumes/Charles/data/vision/torchvision/tno/tno")
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--device", type=str, default='auto')
def test_tno_ablation_cc(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion CC TNO', kwargs)
    opts.present()
    for i in (Path(opts.p) / 'ir').glob("*.png"):
        ir = to_tensor(path_to_gray(i)).to(opts.device)
        vis = path_to_rgb(Path(opts.p) / 'vis' / i.name)
        ir = to_tensor(ir).unsqueeze(0).to(opts.device)
        vis = to_tensor(vis).unsqueeze(0).to(opts.device)
        fused = fusion(ir, vis, kwargs['layer'], debug=False, fusion_method='CC')
        name = Path(opts.p) / 'fused' / 'cpfusion_cc' / i.name
        print(f"Saving {name}")
        save_array_to_img(fused, name, True)



@click.command()
@click.option("--p", type=str, default="/Volumes/Charles/data/vision/torchvision/tno/tno")
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--device", type=str, default='auto')
def test_tno_ablation_max(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion MAX TNO', kwargs)
    opts.present()
    for i in (Path(opts.p) / 'ir').glob("*.png"):
        ir = to_tensor(path_to_gray(i)).to(opts.device)
        vis = path_to_rgb(Path(opts.p) / 'vis' / i.name)
        ir = to_tensor(ir).unsqueeze(0).to(opts.device)
        vis = to_tensor(vis).unsqueeze(0).to(opts.device)
        fused = fusion(ir, vis, kwargs['layer'], debug=False, fusion_method='MAX')
        name = Path(opts.p) / 'fused' / 'cpfusion_max' / i.name
        print(f"Saving {name}")
        save_array_to_img(fused, name, True)

    
@click.command()
@click.option("--p", type=str, default="/Volumes/Charles/data/vision/torchvision/llvip")
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--device", type=str, default='auto')
def test_llvip(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion LLVIP', kwargs)
    opts.present()
    for i in (Path(opts.p) / 'infrared' / 'test').glob("*.jpg"):
        ir = path_to_gray(i)
        vis = path_to_rgb(Path(opts.p) / 'visible' / 'test' / i.name)
        ir = to_tensor(ir).unsqueeze(0).to(opts.device)
        vis = to_tensor(vis).unsqueeze(0).to(opts.device)
        fused = fusion(ir, vis, kwargs['layer'], debug=False)
        name = Path(opts.p) / 'fused' / 'cpfusion' / i.name
        print(f"Saving {name}")
        save_array_to_img(fused, name, True)


@click.command()
@click.option("--p", type=str, default="/Volumes/Charles/data/vision/torchvision/llvip")
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--device", type=str, default='auto')
def test_llvip_ablation_pam(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion Withoud PAM Ablation LLVIP', kwargs)
    opts.present()
    for i in (Path(opts.p) / 'infrared' / 'test').glob("*.jpg"):
        ir = path_to_gray(i)
        vis = path_to_rgb(Path(opts.p) / 'visible' / 'test' / i.name)
        ir = to_tensor(ir).unsqueeze(0).to(opts.device)
        vis = to_tensor(vis).unsqueeze(0).to(opts.device)
        fused = fusion(ir, vis, kwargs['layer'], debug=False, attension='None')
        name = Path(opts.p) / 'fused' / 'cpfusion_wp' / i.name
        print(f"Saving {name}")
        save_array_to_img(fused, name, True)


@click.command()
@click.option("--p", type=str, default="/Volumes/Charles/data/vision/torchvision/llvip")
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--device", type=str, default='auto')
def test_llvip_ablation_max(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion MAX ablation LLVIP', kwargs)
    opts.present()
    for i in (Path(opts.p) / 'infrared' / 'test').glob("*.jpg"):
        ir = path_to_gray(i)
        vis = path_to_rgb(Path(opts.p) / 'visible' / 'test' / i.name)
        ir = to_tensor(ir).unsqueeze(0).to(opts.device)
        vis = to_tensor(vis).unsqueeze(0).to(opts.device)
        fused = fusion(ir, vis, kwargs['layer'], debug=False, fusion_method='MAX')
        name = Path(opts.p) / 'fused' / 'cpfusion_max' / i.name
        print(f"Saving {name}")
        save_array_to_img(fused, name, True)



@click.command()
@click.option("--p", type=str, default="/Volumes/Charles/data/vision/torchvision/llvip")
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--device", type=str, default='auto')
def test_llvip_ablation_cc(**kwargs) -> None:
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion CC ablation LLVIP', kwargs)
    opts.present()
    for i in (Path(opts.p) / 'infrared' / 'test').glob("*.jpg"):
        ir = path_to_gray(i)
        vis = path_to_rgb(Path(opts.p) / 'visible' / 'test' / i.name)
        ir = to_tensor(ir).unsqueeze(0).to(opts.device)
        vis = to_tensor(vis).unsqueeze(0).to(opts.device)
        fused = fusion(ir, vis, kwargs['layer'], debug=False, fusion_method='CC')
        name = Path(opts.p) / 'fused' / 'cpfusion_cc' / i.name
        print(f"Saving {name}")
        save_array_to_img(fused, name, True)



@click.command()
@click.option("--layer", type=int, default=4)
@click.option("--msd_method", type=str, default=['Laplacian','Contrust'][0])
@click.option("--device", type=str, default='auto')
def ablation(**kwargs):
    kwargs['device'] = get_device(kwargs['device'])
    opts = Options('CPFusion Ablation', kwargs)

    image_index = 52
    ir = path_to_gray(f'/Volumes/Charles/data/vision/torchvision/tno/tno/ir/{image_index}.png')
    vis = path_to_rgb(f'/Volumes/Charles/data/vision/torchvision/tno/tno/vis/{image_index}.png')
    ir = to_tensor(ir).unsqueeze(0).to(opts.device)
    vis = to_tensor(vis).unsqueeze(0).to(opts.device)
    fused_cc_tno = fusion(ir, vis, kwargs['layer'], fusion_method='CC', debug=False)
    fused_max_tno = fusion(ir, vis, kwargs['layer'], fusion_method='MAX', debug=False)
    fused_cc_max_tno = fusion(ir, vis, kwargs['layer'], fusion_method='CC+MAX', debug=False)
    fused_without_pam_tno = fusion(ir, vis, kwargs['layer'], attension='None', debug=False)
    fused_pam_tno = fusion(ir, vis, kwargs['layer'], attension='DSimAM', debug=False)

    llvip_index = 190190
    ir_llvip = path_to_gray(f'/Volumes/Charles/data/vision/torchvision/llvip/infrared/test/{llvip_index}.jpg')
    vis_llvip = path_to_rgb(f'/Volumes/Charles/data/vision/torchvision/llvip/visible/test/{llvip_index}.jpg')
    ir_llvip = to_tensor(ir_llvip).unsqueeze(0).to(opts.device)
    vis_llvip = to_tensor(vis_llvip).unsqueeze(0).to(opts.device)
    fused_cc_llvip = fusion(ir_llvip, vis_llvip, kwargs['layer'], fusion_method='CC', debug=False)
    fused_max_llvip = fusion(ir_llvip, vis_llvip, kwargs['layer'], fusion_method='MAX', debug=False)
    fused_cc_max_llvip = fusion(ir_llvip, vis_llvip, kwargs['layer'], fusion_method='CC+MAX', debug=False)
    fused_without_pam_llvip = fusion(ir_llvip, vis_llvip, kwargs['layer'], attension='None', debug=False)
    fused_pam_llvip = fusion(ir_llvip, vis_llvip, kwargs['layer'], attension='DSimAM', debug=False)

    glance(
            [fused_cc_tno, fused_max_tno, fused_cc_max_tno]+\
            [fused_cc_llvip, fused_max_llvip, fused_cc_max_llvip],
            shape = (2,3),
            each_save = True,
            each_save_dir = "./glance_outputs/ablantion/1"
        )

    glance(
            [fused_without_pam_tno, fused_pam_tno]+\
            [fused_without_pam_llvip, fused_pam_llvip],
            shape = (2,2),
            each_save = True,
            each_save_dir = "./glance_outputs/ablantion/2"
        )

    

if __name__ == '__main__':
    # main()
    # test_tno()
    test_llvip()
    # ablation()
    # test_tno_ablation_pam()
    # test_tno_ablation_cc()
    # test_tno_ablation_max()
    # test_llvip_ablation_pam()
    # test_llvip_ablation_cc()
    # test_llvip_ablation_max()
