import click
import sqlite3
from cslib.utils import glance, path_to_gray, path_to_rgb, save_array_to_img, to_tensor
from cslib.datasets.fusion import TNO, LLVIP
from utils import *
from model import *
from pathlib import Path
from cslib.metrics import fusion

def summarize_metrics(db_path, fuse_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取所有指标名称
    cursor.execute('''
        SELECT DISTINCT metric FROM metrics
    ''')
    metrics = [row[0] for row in cursor.fetchall()]

    print(f"\nSummary for fuse_name: {fuse_name}")
    for metric in metrics:
        cursor.execute('''
            SELECT AVG(value) FROM metrics
            WHERE fuse_name = ? AND metric = ?
        ''', (fuse_name, metric))
        avg_value = cursor.fetchone()[0]
        print(f"  Metric: {metric.upper()}, Average Value: {avg_value:.4f}")

    conn.close()

@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--fuse_path", type=click.Path(exists=True), required=True)
@click.option("--layer", type=int, default=4)
@click.option("--metrics_path", type=click.Path(exists=True), required=True)
@click.option("--fuse_name", type=str, required=True)
def main(**kwargs):
    conn, cursor = creat_database(kwargs['metrics_path'])
    dataset = LLVIP(root=kwargs['dataset_path'], transform=None, download=True, train=False)
    for i in range(len(dataset)):
        if i%50 != 0:
            continue
        ir,vis = dataset[i]
        fused = Path(kwargs['fuse_path']) / Path(ir).name
        ir_img = to_tensor(path_to_rgb(ir)).unsqueeze(0)
        vis_img = to_tensor(path_to_rgb(vis)).unsqueeze(0)
        fused_img = to_tensor(path_to_rgb(fused)).unsqueeze(0)
        # glance([ir_img, vis_img, fused_img])
        
        for metric_name, metric_func in md.items():
            value = metric_func(ir_img, vis_img, fused_img).item()
            print(f"Image: {Path(ir).name}, Fuse: {kwargs['fuse_name']}, Metric: {metric_name.upper()}, Value: {value:.4f}")
            cursor.execute('''
                INSERT OR REPLACE INTO metrics (id, fuse_name, metric, value)
                VALUES (?, ?, ?, ?)
            ''', (Path(ir).name, kwargs['fuse_name'], metric_name, value))
            conn.commit()
    conn.close()

if __name__ == '__main__':
    main()
