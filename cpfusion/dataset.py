import click
from cslib.datasets.fusion import TNO, LLVIP
    
@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
def test_tno(**kwargs):
    tno = TNO(root=kwargs['dataset_path'], img_type='lwir', mode='both', transform=None, download=True)
    print('tno (lwir,pair+seq)',len(tno))
    # tno (lwir,pair+seq) 235
    
@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
def test_llvip(**kwargs):
    llvip_test = LLVIP(root=kwargs['dataset_path'], download=True, train=False)
    print('llvip_test',len(llvip_test))
    llvip_train = LLVIP(root=kwargs['dataset_path'], download=True, train=True)
    print('llvip_train',len(llvip_train))

    # llvip_test 3463
    # llvip_train 12025

@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--train_size", type=float, default=0.6)
@click.option("--test_size", type=float, default=0.2)
@click.option("--val_size", type=float, default=0.2)
@click.option("--llvip_val_size", type=float, default=0.2)
@click.option("--seed", type=int, default=42)
def test(**kwargs):
    # TNO
    tno = TNO(root=kwargs['dataset_path'], img_type='lwir', mode='both', transform=None, download=True)
    train_size = int(kwargs['train_size'] * len(tno))
    val_size = int(kwargs['val_size'] * len(tno))
    test_size = len(tno) - train_size - val_size
    tno_train, tno_val, tno_test = random_split(tno,[train_size,val_size,test_size],
            generator=torch.Generator().manual_seed(kwargs['seed']))
    
    # LLVIP
    temp = LLVIP(root=kwargs['dataset_path'], download=True, train=True)
    llvip_val_size = int(kwargs['llvip_val_size'] * len(temp))
    llvip_train_size = len(temp) - llvip_val_size
    llvip_val, llvip_train = random_split(temp,[llvip_val_size,llvip_train_size],
            generator=torch.Generator().manual_seed(kwargs['seed']))
    llvip_test = LLVIP(root=kwargs['dataset_path'], download=True, train=False)

    # Datasets
    train_dataset = ConcatDataset([tno_train,llvip_train])
    test_dataset = ConcatDataset([tno_test,llvip_test])
    val_dataset = ConcatDataset([tno_val,llvip_val])

    print(len(train_dataset),len(test_dataset),len(val_dataset))
    # 9761 3510 2452

if __name__ == '__main__':
    from torch.utils.data import ConcatDataset
    from torch.utils.data import random_split
    import torch
    test()
    # test_tno()
    # test_llvip()