import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('MacOSX')
import numpy as np
from utils import calculate_SF
from cslib.metrics.fusion import ir,vis

def visualize_wcc_process(ir_tensor, vis_tensor, save_path=None):
    """
    可视化WCC模块的处理过程和中间结果
    Args:
        ir_tensor: 红外图像张量 (1, 1, H, W)
        vis_tensor: 可见光图像张量 (1, 1, H, W)
        save_path: 结果保存路径（可选）
    """
    # 确保输入为单通道
    if ir_tensor.dim() == 3:
        ir_tensor = ir_tensor.unsqueeze(0)
    if vis_tensor.dim() == 3:
        vis_tensor = vis_tensor.unsqueeze(0)
    
    # 计算相关系数和权重
    with torch.no_grad():
        # 计算相关系数
        ir_mean = ir_tensor.mean(dim=(2, 3), keepdim=True)
        vis_mean = vis_tensor.mean(dim=(2, 3), keepdim=True)
        
        cov_XY = torch.mean((ir_tensor - ir_mean) * (vis_tensor - vis_mean))
        var_ir = torch.var(ir_tensor, dim=(2, 3), keepdim=True)
        var_vis = torch.var(vis_tensor, dim=(2, 3), keepdim=True)
        
        epsilon = 1e-8
        V_CC = cov_XY / torch.sqrt(var_ir * var_vis + epsilon)
        W_CC = torch.where(V_CC > 0, 1 - torch.exp(-V_CC), torch.exp(V_CC) - 1)
        
        # 基础层融合结果
        weight = torch.clamp(W_CC, 0.0, 1.0)
        fused_base = weight * ir_tensor + (1 - weight) * vis_tensor
        
        # 细节层融合（简化版）
        sf_ir = calculate_SF(ir_tensor)
        sf_vis = calculate_SF(vis_tensor)
        alpha = sf_ir / (sf_ir + sf_vis + epsilon)
        fused_detail = alpha * ir_tensor + (1 - alpha) * vis_tensor

    # 转换为numpy用于可视化
    ir_np = ir_tensor.squeeze().cpu().numpy()
    vis_np = vis_tensor.squeeze().cpu().numpy()
    weight_np = weight.squeeze().cpu().numpy()
    fused_base_np = fused_base.squeeze().cpu().numpy()
    fused_detail_np = fused_detail.squeeze().cpu().numpy()

    breakpoint()

    # 创建可视化图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(ir_np, cmap='gray')
    axes[0, 0].set_title('Infrared Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(vis_np, cmap='gray')
    axes[0, 1].set_title('Visible Image')
    axes[0, 1].axis('off')
    
    # 权重图
    im = axes[0, 2].imshow(weight_np, cmap='viridis')
    axes[0, 2].set_title(f'WCC Weight: {weight_np.mean():.3f}')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # 融合结果
    axes[1, 0].imshow(fused_base_np, cmap='gray')
    axes[1, 0].set_title('Base Layer Fusion')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(fused_detail_np, cmap='gray')
    axes[1, 1].set_title('Detail Layer Fusion')
    axes[1, 1].axis('off')
    
    # 统计信息
    axes[1, 2].text(0.1, 0.8, f'Correlation Coef: {V_CC.item():.3f}', fontsize=12)
    axes[1, 2].text(0.1, 0.6, f'Weight Value: {weight_np.mean():.3f}', fontsize=12)
    axes[1, 2].text(0.1, 0.4, f'IR Variance: {var_ir.item():.3f}', fontsize=12)
    axes[1, 2].text(0.1, 0.2, f'VIS Variance: {var_vis.item():.3f}', fontsize=12)
    axes[1, 2].set_title('Statistical Information')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")
    
    plt.show()
    
    return {
        'correlation_coef': V_CC.item(),
        'weight_value': weight_np.mean(),
        'ir_variance': var_ir.item(),
        'vis_variance': var_vis.item()
    }

# 使用示例
if __name__ == '__main__':
    # 运行可视化
    results = visualize_wcc_process(ir, vis, save_path='wcc_debug.png')
    
    print("调试结果:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")