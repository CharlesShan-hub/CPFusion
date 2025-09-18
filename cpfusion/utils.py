import torch 

__all__ = [
    'msd_align', 
    'msd_resample', 
    'correlation_coefficient_weights', 
    '_base_layer_fuse',
    '_detail_layer_fuse',
]

def msd_align(images):    
    def up_sample(img, target_shape):
        batch_size, channels, height, width = img.size()
        padded_img = torch.zeros(batch_size, channels, 2 * height, 2 * width, device=img.device)
        padded_img[:, :, ::2, ::2] = img
        padded_img[:, :, 1::2, ::2] = img
        padded_img[:, :, ::2, 1::2] = img
        padded_img[:, :, 1::2, 1::2] = img
        return padded_img[:,:,:target_shape[-2],:target_shape[-1]]
    
    if len(images) == 1:
        return images
    
    queue = [images[-1]]
    for i in range(len(images) - 1, 0, -1):
        for _ in range(len(images)-i):
            queue.insert(0,up_sample(queue.pop(), images[i-1].shape))
        queue.insert(0,images[i-1])
    return queue

def msd_resample(tensor):
    images = list(torch.split(tensor, 1, dim=1))
    for i in range(len(images)-1): # 0, 1, 2, 3
        for _ in range(len(images)-1-i):
            images[-1-i] = images[-1-i][:,:, ::2, ::2]
    return images

def correlation_coefficient_weights(X, Y):
    # correlation (-n, n)
    cov_XY = torch.mean((X - X.mean(dim=(2, 3), keepdim=True)) * (Y - Y.mean(dim=(2, 3), keepdim=True)))

    # variation (n)
    var_X = torch.var(X, dim=(2, 3), keepdim=True)
    var_Y = torch.var(Y, dim=(2, 3), keepdim=True)
    
    # correlation coefficient (-1, 1)
    V_CC = cov_XY / torch.sqrt(var_X * var_Y) 
    
    # weights
    # W_CC = torch.where(V_CC > 0, 1 - torch.exp(-V_CC), torch.exp(V_CC) - 1)
    W_CC = torch.where(V_CC > 0, 1 - torch.exp(-V_CC), torch.exp(V_CC) - 1)
    
    return W_CC

def _base_layer_fuse(X, Y, wcc):
    weight = 1.5 ** ((wcc + 1) / 2)
    # weight = torch.clamp(weight, 0, 1) # Remove this
    fused = torch.zeros_like(X)
    weighted_X = weight * X
    weighted_Y = (2 - weight) * Y# Changed to this
    fused[:, ::2, :, :] = torch.max(X, Y)[:, ::2, :, :]
    fused[:,1::2, :, :] = (weighted_X + weighted_Y)[:,1::2, :, :]/2 # Changed to this
    # glance([_c(fused[:,0:1,:,:]),_c(fused[:,1:2,:,:]),_c(fused[:,2:3,:,:]),_c(fused[:,3:4,:,:])])
    # glance([_c(torch.max(X, Y)[:,0:1,:,:]), _c((weighted_X + weighted_Y)[:,0:1,:,:]), _c(X[:,0:1,:,:]), _c(Y[:,0:1,:,:])])
    # breakpoint()
    return fused
    # return fused.mean(dim=1,keepdim=True)

def calculate_SF(X):
    # Ensure the input is a 2D tensor
    assert X.dim() >= 2, "Input must be at least a 2D tensor."

    # Perform 2D FFT
    fftimage = torch.fft.fft2(X,dim=(-2,-1))

    # Compute the amplitude spectrum
    amplitudespectrum = torch.abs(fftimage)

    # Calculate the mean of the amplitude spectrum
    sf = torch.mean(amplitudespectrum,dim=(-2,-1))

    return sf

def divide_tensor_into_blocks(tensor, block_size=8):
    """
    将输入的张量划分为 8x8 的块。
    
    参数:
        tensor (torch.Tensor): 输入张量，形状为 (B, C, H, W)。
        block_size (int): 每个块的大小，默认为 8。
    
    返回:
        torch.Tensor: 划分后的块，形状为 (B, C, num_blocks_h, num_blocks_w, block_size, block_size)。
    """
    # 获取输入张量的形状
    B, C, H, W = tensor.shape
    
    # 确保 H 和 W 是 block_size 的倍数，如果不是则填充到合适的大小
    if H % block_size != 0 or W % block_size != 0:
        pad_h = block_size - (H % block_size)
        pad_w = block_size - (W % block_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        H, W = tensor.shape[2], tensor.shape[3]
    
    # 使用 unfold 方法将 H 和 W 维度划分为 block_size 的块
    num_blocks_h = H // block_size
    num_blocks_w = W // block_size
    
    unfolded_tensor = tensor.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    
    # 调整形状以方便后续处理
    result = unfolded_tensor.contiguous().view(B, C, num_blocks_h, num_blocks_w, block_size, block_size)
    
    return result

def restore_tensor_from_blocks(blocks,original_shape):
    """
    将划分后的块重新组合为原始张量。
    
    参数:
        blocks (torch.Tensor): 划分后的块，形状为 (B, C, num_blocks_h, num_blocks_w, block_size, block_size)。
        original_shape (tuple): 原始张量的形状 (B, C, H, W)。
    
    返回:
        torch.Tensor: 恢复后的张量，形状为 (B, C, H, W)。
    """
    # 获取块张量的形状
    B, C, num_blocks_h, num_blocks_w, block_size, _ = blocks.shape
    
    # 计算恢复后的张量的形状
    H = num_blocks_h * block_size
    W = num_blocks_w * block_size
    
    # 如果原始张量的 H 或 W 不是块大小的整数倍，说明之前有填充，需要裁剪
    pad_h = H - original_shape[2]
    pad_w = W - original_shape[3]
    
    # 将块重新组合为张量
    restored_tensor = blocks.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    
    # 如果有填充，则裁剪掉填充的部分
    if pad_h > 0 or pad_w > 0:
        restored_tensor = restored_tensor[:, :, :original_shape[2], :original_shape[3]]
    
    return restored_tensor


# def split_img(X, n=3):
#     def mirror_pad_if_odd_and_unfold(tensor):
#         # padding
#         pad_height = 0 if (tensor.shape[-2] % 2 == 0) else 1
#         pad_width = 0 if (tensor.shape[-1] % 2 == 0) else 1
#         tensor = F.pad(tensor, (0, pad_width, 0, pad_height),mode='constant')#, mode='reflect')
        
#         # unfold
#         H, W = tensor.shape[-2]//2, tensor.shape[-1]//2
#         return torch.cat([
#             tensor[..., :H, :W], # Top - Left
#             tensor[..., :H,-W:], # Top - Right
#             tensor[...,-H:, :W], # Bottom - Left
#             tensor[...,-H:,-W:], # Bottom - Right
#         ],dim=-3), (pad_height,pad_width)
    
#     paddings = []
#     for _ in range(n):
#         X, pad = mirror_pad_if_odd_and_unfold(X)
#         paddings.append(pad)
#     return X,paddings

# def merge_img(X, paddings, n=3):
#     def fold(tensor,pad):
#         (B,C,H,W) = tensor.shape
#         C//=4
#         padded = torch.zeros(size=(B,C,H*2,W*2))
#         padded[:,:,:H,:W] = tensor[:,0*C:1*C,:,:]
#         padded[:,:,:H,-W:] = tensor[:,1*C:2*C,:,:]
#         padded[:,:,-H:,:W] = tensor[:,2*C:3*C,:,:]
#         padded[:,:,-H:,-W:] = tensor[:,3*C:4*C,:,:]
#         return padded[:,:,:H*2-pad[0],:W*2-pad[1]]

#     for _ in range(n):
#         X = fold(X, paddings.pop())
#     return X

def _detail_layer_fuse(X,Y):
    patchx = divide_tensor_into_blocks(X)
    patchy = divide_tensor_into_blocks(Y)
    # print(patchx.shape)
    SF_X = calculate_SF(patchx)
    # print(SF_X.shape)
    SF_Y = calculate_SF(patchy)
    alpha = (SF_X / (SF_X + SF_Y)).unsqueeze(-1).unsqueeze(-1)
    # print(alpha.shape)
    patchf = alpha * patchx + (1 - alpha) * patchy
    # print(patchf.shape)
    # return restore_tensor_from_blocks(patchf, X.shape).mean(dim=1,keepdim=True)
    return restore_tensor_from_blocks(patchf, X.shape)

# def test_split_and_merge():
#     a1 = torch.rand(1, 1, 16, 16)
#     a2 = torch.rand(2, 1, 16, 15)
#     a3 = torch.rand(1, 3, 15, 14)
#     a4 = torch.rand(8, 3, 16, 17)

#     b1,p1 = split_img(a1)
#     b2,p2 = split_img(a2)
#     b3,p3 = split_img(a3)
#     b4,p4 = split_img(a4)

#     c1 = merge_img(b1,p1)
#     c2 = merge_img(b2,p2)
#     c3 = merge_img(b3,p3)
#     c4 = merge_img(b4,p4)

#     assert a1.equal(c1)
#     assert a2.equal(c2)
#     assert a3.equal(c3)
#     assert a4.equal(c4)

def test_split_and_merge():
    # 创建一个随机张量
    B, C, H, W = 2, 3, 24, 32
    original_tensor = torch.randn(B, C, H, W)
    print("原始张量形状:", original_tensor.shape)
    
    # 划分为 8x8 的块
    block_size = 8
    blocks = divide_tensor_into_blocks(original_tensor, block_size)
    print("划分后的块形状:", blocks.shape)
    
    # 恢复为原始张量
    restored_tensor = restore_tensor_from_blocks(blocks, original_tensor.shape)
    print("恢复后的张量形状:", restored_tensor.shape)
    
    # 验证恢复后的张量是否与原始张量一致
    print("恢复后的张量是否与原始张量一致:", original_tensor.equal(restored_tensor))

if __name__ == '__main__':
    test_split_and_merge()