import click
from cslib.utils import *
import matplotlib.pyplot as plt
import numpy as np
from cslib.utils.config import Options
import matplotlib
# other backends:
# https://matplotlib.org/stable/users/explain/figure/backends.html
matplotlib.use('macosx')

# 设置中文字体支持
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义默认文件路径
cc1_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/cc_max_ablation/20/3.png"
max1_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/cc_max_ablation/20/4.png"
cc_max1_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/cc_max_ablation/20/2.png"
cc2_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/cc_max_ablation/190822/3.png"
max2_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/cc_max_ablation/190822/4.png"
cc_max2_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/cc_max_ablation/190822/2.png"
cc3_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/cc_max_ablation/00388/3.png"
max3_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/cc_max_ablation/00388/4.png"
cc_max3_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/cc_max_ablation/00388/2.png"
cc4_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/cc_max_ablation/00947N/3.png"
max4_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/cc_max_ablation/00947N/4.png"
cc_max4_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/cc_max_ablation/00947N/2.png"
# default_mode = 'cc'
# row_zoom_regions = [
#     (434, 353, 100, 80),    # 第一行图像的放大区域
#     (120, 250, 100, 80),    # 第二行图像的放大区域
#     (446, 130, 100, 80),    # 第三行图像的放大区域
#     (384, 316, 90, 75)      # 第四行图像的放大区域
# ]

pam1_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/pam_ablation/20/2.png"
nopam1_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/pam_ablation/20/3.png"
pam2_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/pam_ablation/190822/2.png"
nopam2_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/pam_ablation/190822/3.png"
pam3_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/pam_ablation/00388/2.png"
nopam3_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/pam_ablation/00388/3.png"
pam4_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/pam_ablation/00947N/2.png"
nopam4_dir = "/Users/kimshan/Public/project/CPFusion/assets/glance_outputs/pam_ablation/00947N/3.png"
default_mode = 'pam'
row_zoom_regions = [
    (199, 397, 50, 40),     # 第一行图像的放大区域
    (108, 450, 100, 80),    # 第二行图像的放大区域
    (173, 180, 100, 80),    # 第三行图像的放大区域
    (251, 240, 90, 75)      # 第四行图像的放大区域
]

def auto_resize(image, width):
    """按比例调整图像大小，保持宽高比不变"""
    # 获取原始图像的尺寸
    h, w = image.shape[:2]
    # 计算新的高度，保持宽高比
    height = int((h / w) * width)
    # 使用双线性插值调整图像大小
    # 创建一个新的空白图像用于存储调整后的图像
    resized = np.zeros((height, width, 3), dtype=image.dtype)
    # 计算缩放因子
    h_scale = h / height
    w_scale = w / width
    # 双线性插值实现
    for i in range(height):
        for j in range(width):
            # 计算在原始图像中的坐标
            orig_i = i * h_scale
            orig_j = j * w_scale
            
            # 找到周围的四个像素
            i0 = int(orig_i)
            i1 = min(i0 + 1, h - 1)
            j0 = int(orig_j)
            j1 = min(j0 + 1, w - 1)
            
            # 计算权重
            di = orig_i - i0
            dj = orig_j - j0
            
            # 双线性插值
            for c in range(3):  # 处理RGB三个通道
                resized[i, j, c] = (
                    (1 - di) * (1 - dj) * image[i0, j0, c] +
                    di * (1 - dj) * image[i1, j0, c] +
                    (1 - di) * dj * image[i0, j1, c] +
                    di * dj * image[i1, j1, c]
                )
    
    return resized

def add_zoom_region(image, x, y, a, b, zoom_factor=2):
    """为图像添加放大区域
    参数:
        image: 输入图像
        x, y: 放大区域左上角坐标
        a, b: 放大区域的宽和高
        zoom_factor: 放大倍数
    返回:
        带有放大区域的新图像
    """
    h, w = image.shape[:2]
    
    # 创建一个新图像，用于放置原图和放大区域
    new_h = h
    new_w = w
    
    # 计算放大区域的大小
    zoom_h = int(b * zoom_factor)
    zoom_w = int(a * zoom_factor)
    
    # 确保放大区域不会超出图像边界
    x_end = min(x + a, w)
    y_end = min(y + b, h)
    
    # 裁剪要放大的区域
    roi = image[y:y_end, x:x_end].copy()
    
    # 调整roi的大小（放大）
    zoomed_roi = np.zeros((zoom_h, zoom_w, 3), dtype=image.dtype)
    
    # 双线性插值放大
    h_scale = (y_end - y) / zoom_h
    w_scale = (x_end - x) / zoom_w
    
    for i in range(zoom_h):
        for j in range(zoom_w):
            # 计算在原始roi中的坐标
            orig_i = int(i * h_scale)
            orig_j = int(j * w_scale)
            
            # 确保不越界
            orig_i = min(orig_i, roi.shape[0] - 1)
            orig_j = min(orig_j, roi.shape[1] - 1)
            
            # 复制像素值
            zoomed_roi[i, j] = roi[orig_i, orig_j]
    
    # 在原图上绘制矩形框
    image_with_box = image.copy()
    
    # 绘制矩形框（使用红色）
    color = [255, 0, 0]  # 红色
    thickness = max(1, int(h / 200))  # 根据图像高度调整线宽
    
    # 绘制上、下边
    image_with_box[y:y+thickness, x:x_end] = color
    image_with_box[y_end-thickness:y_end, x:x_end] = color
    # 绘制左、右边
    image_with_box[y:y_end, x:x+thickness] = color
    image_with_box[y:y_end, x_end-thickness:x_end] = color
    
    # 将放大区域放置在右下角
    # 计算放置位置
    zoom_x = w - zoom_w - 10
    zoom_y = h - zoom_h - 10
    
    # 确保放置位置有效
    zoom_x = max(0, zoom_x)
    zoom_y = max(0, zoom_y)
    
    # 复制放大区域到新位置
    result = image_with_box.copy()
    result[zoom_y:zoom_y+zoom_h, zoom_x:zoom_x+zoom_w] = zoomed_roi
    
    # 在放大区域周围绘制边框
    result[zoom_y:zoom_y+thickness, zoom_x:zoom_x+zoom_w] = color
    result[zoom_y+zoom_h-thickness:zoom_y+zoom_h, zoom_x:zoom_x+zoom_w] = color
    result[zoom_y:zoom_y+zoom_h, zoom_x:zoom_x+thickness] = color
    result[zoom_y:zoom_y+zoom_h, zoom_x+zoom_w-thickness:zoom_x+zoom_w] = color
    
    return result


@click.command()
@click.option('--mode', default=default_mode, help='可视化模式: cc 或 pam')
@click.option('--cc1', default=cc1_dir, help='CC方法第一张图像路径')
@click.option('--max1', default=max1_dir, help='MAX方法第一张图像路径')
@click.option('--cc_max1', default=cc_max1_dir, help='CC+MAX方法第一张图像路径')
@click.option('--cc2', default=cc2_dir, help='CC方法第二张图像路径')
@click.option('--max2', default=max2_dir, help='MAX方法第二张图像路径')
@click.option('--cc_max2', default=cc_max2_dir, help='CC+MAX方法第二张图像路径')
@click.option('--cc3', default=cc3_dir, help='CC方法第三张图像路径')
@click.option('--max3', default=max3_dir, help='MAX方法第三张图像路径')
@click.option('--cc_max3', default=cc_max3_dir, help='CC+MAX方法第三张图像路径')
@click.option('--cc4', default=cc4_dir, help='CC方法第四张图像路径')
@click.option('--max4', default=max4_dir, help='MAX方法第四张图像路径')
@click.option('--cc_max4', default=cc_max4_dir, help='CC+MAX方法第四张图像路径')
@click.option('--pam1', default=pam1_dir, help='PAM方法第一张图像路径')
@click.option('--nopam1', default=nopam1_dir, help='无 PAM方法第一张图像路径')
@click.option('--pam2', default=pam2_dir, help='PAM方法第二张图像路径')
@click.option('--nopam2', default=nopam2_dir, help='无 PAM方法第二张图像路径')
@click.option('--pam3', default=pam3_dir, help='PAM方法第三张图像路径')
@click.option('--nopam3', default=nopam3_dir, help='无 PAM方法第三张图像路径')
@click.option('--pam4', default=pam4_dir, help='PAM方法第四张图像路径')
@click.option('--nopam4', default=nopam4_dir, help='无 PAM方法第四张图像路径')
@click.option('--margin', default=20)
# @click.option('--save_path', default='./assets/glance_outputs/cc_max_ablation/res1.png', help='保存图像的路径，默认为不保存')
@click.option('--save_path', default='./assets/glance_outputs/cc_max_ablation/res2.png', help='保存图像的路径，默认为不保存')
def main(**kwargs):
    opts = Options('Draw Details', kwargs)
    
    if opts.mode == 'cc':
        # 加载图像
        cc1 = to_numpy(path_to_rgb(opts.cc1))
        width = cc1.shape[1]
        # 加载和调整图像大小
        images = [cc1]
        images.append(auto_resize(to_numpy(path_to_rgb(opts.cc_max1)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.max1)), width))
        
        images.append(auto_resize(to_numpy(path_to_rgb(opts.cc2)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.cc_max2)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.max2)), width))
        
        images.append(auto_resize(to_numpy(path_to_rgb(opts.cc3)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.cc_max3)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.max3)), width))
        
        images.append(auto_resize(to_numpy(path_to_rgb(opts.cc4)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.cc_max4)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.max4)), width))
        
        # 为每行的3张图像添加相同位置的放大区域
        for i in range(len(images)):
            # 计算当前图像属于哪一行（每3张图像为一行）
            row_index = i // 3
            if row_index < len(row_zoom_regions):
                # 获取当前行的放大区域坐标
                x, y, a, b = row_zoom_regions[row_index]
                # 根据图像大小调整缩放区域坐标
                img_h, img_w = images[i].shape[:2]
                # 确保坐标在图像范围内
                x = min(x, img_w - 1)
                y = min(y, img_h - 1)
                a = min(a, img_w - x)
                b = min(b, img_h - y)
                
                # 添加放大区域
                images[i] = add_zoom_region(images[i], x, y, a, b)

        image = np.ones(shape=(images[0].shape[0]+images[3].shape[0]+images[6].shape[0]+images[9].shape[0]+3*opts.margin,3*cc1.shape[1]+2*opts.margin,3))

        w = 0
        h = 0
        for i in range(4):
            for j in range(3):
                m = images[i*3+j]
                image[h: h+m.shape[0],w: w+m.shape[1],:] = m
                w += (m.shape[1]+opts.margin)
            w = 0
            h += (m.shape[0]+opts.margin)

        plt.imshow(image)
        plt.axis('off')  # 隐藏坐标轴
        plt.tight_layout()  # 自动调整子图参数，防止标签被裁剪
        if opts.save_path:
            plt.savefig(opts.save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {opts.save_path}")
        plt.show()
        
    elif opts.mode == 'pam':
        pam1 = to_numpy(path_to_rgb(opts.pam1))
        width = pam1.shape[1]
        images = [pam1]
        images.append(auto_resize(to_numpy(path_to_rgb(opts.nopam1)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.pam2)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.nopam2)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.pam3)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.nopam3)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.pam4)), width))
        images.append(auto_resize(to_numpy(path_to_rgb(opts.nopam4)), width))
        
        # 为每行的2张图像添加相同位置的放大区域
        for i in range(len(images)):
            # 计算当前图像属于哪一行（每2张图像为一行）
            row_index = i // 2
            if row_index < len(row_zoom_regions):
                # 获取当前行的放大区域坐标
                x, y, a, b = row_zoom_regions[row_index]
                # 根据图像大小调整缩放区域坐标
                img_h, img_w = images[i].shape[:2]
                # 确保坐标在图像范围内
                x = min(x, img_w - 1)
                y = min(y, img_h - 1)
                a = min(a, img_w - x)
                b = min(b, img_h - y)
                
                # 添加放大区域
                images[i] = add_zoom_region(images[i], x, y, a, b)
            
        image = np.ones(shape=(images[0].shape[0]+images[2].shape[0]+images[4].shape[0]+images[6].shape[0]+3*opts.margin,2*pam1.shape[1]+opts.margin,3))

        w = 0
        h = 0
        for i in range(4):
            for j in range(2):
                m = images[i*2+j]
                image[h: h+m.shape[0],w: w+m.shape[1],:] = m
                w += (m.shape[1]+opts.margin)
            w = 0
            h += (m.shape[0]+opts.margin)
        plt.imshow(image)
        plt.axis('off')  # 隐藏坐标轴
        plt.tight_layout()  # 自动调整子图参数，防止标签被裁剪
        if opts.save_path:
            plt.savefig(opts.save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {opts.save_path}")
        plt.show()

    else:
        print(f"未知模式: {opts.mode}")


if __name__ == '__main__':
    main()