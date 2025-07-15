import torch
import torch.nn as nn

__all__ = [
    'DSimAMBlock'
]

class DSimAMBlock(torch.nn.Module):
    '''
    detail enhancement and feature highlighting module
    detail + SimAMBlock = DSimAMBlock
    '''
    def __init__(self):
        super(DSimAMBlock, self).__init__()
    
    def forward(self, X):
        # 计算图像的平均像素值
        mean_X = X.mean(dim=(2, 3), keepdim=True)
        
        # 计算每个像素与平均值的差的平方
        d = (X - mean_X) ** 2
        
        # 计算激活函数
        func = 1 / (1 + torch.exp(-d)) + 0.5
        
        # 应用激活函数到像素差异值，得到增强图像
        X_enhance = X * func
        
        return X_enhance

def main():
    # 假设输入的红外和可见光图像
    input_ir = torch.randn(1, 3, 256, 256)
    input_vis = torch.randn(1, 3, 256, 256)
    
    # 创建DSimAMBlock实例
    model = DSimAMBlock()
    
    # 分别增强红外和可见光图像的细节层
    enhanced_ir = model(input_ir)
    enhanced_vis = model(input_vis)
    
    # 计算基于相关系数的权重
    # weights_CC = model.calculate_correlation_coefficient_weights(enhanced_ir, enhanced_vis)
    
    print("Enhanced Infrared Image Detail Layer:", enhanced_ir.shape)
    print("Enhanced Visible Light Image Detail Layer:", enhanced_vis.shape)
    # print("Correlation Coefficient Weights:", weights_CC.shape)


if __name__ == '__main__':
    main()
