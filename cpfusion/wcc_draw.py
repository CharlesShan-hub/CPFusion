import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, use
use("macosx")

# 设置学术图表样式
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
rcParams['mathtext.fontset'] = 'stix'

# 定义函数
def vcc_to_wcc(vcc):
    return np.where(vcc > 0, 1 - np.exp(-vcc), np.exp(vcc) - 1)

def wcc_to_final_weight(wcc):
    return 1.5 ** ((wcc + 1) / 2)

# 生成数据
vcc = np.linspace(-1, 1, 500)
wcc = vcc_to_wcc(vcc)
final_weight = wcc_to_final_weight(wcc)

# 创建图表
plt.figure(figsize=(8, 6), dpi=300)

# 绘制两条曲线
plt.plot(vcc, wcc, 'b-', linewidth=2.5, label=r'$W^{CC}(V^{CC})$')
plt.plot(vcc, final_weight, 'r-', linewidth=2.5, label=r'$W = 1.5^{(W^{CC} + 1)/2}$')

# 标记关键点
critical_points = [-1, -0.5, 0, 0.5, 1]
for x in critical_points:
    y_wcc = vcc_to_wcc(x)
    y_final = wcc_to_final_weight(y_wcc)
    plt.scatter(x, y_wcc, c='blue', s=80, zorder=5)
    plt.scatter(x, y_final, c='red', s=80, zorder=5)
    plt.text(x, y_wcc+0.05, f'({x:.1f}, {y_wcc:.2f})', 
             ha='center', fontsize=10, color='blue')
    plt.text(x, y_final+0.05, f'({x:.1f}, {y_final:.2f})', 
             ha='center', fontsize=10, color='red')

# 设置标题和公式说明
plt.title('Weight Transformation Functions\n', fontsize=16, pad=10)
plt.text(0.5, 0.92, 
         r'$W^{CC} = 1 - e^{-V^{CC}}\quad (V^{CC}>0)$',
         fontsize=12, ha='center', va='bottom', 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.text(0.5, 0.87, 
         r'$W^{CC} = e^{V^{CC}} - 1\quad (V^{CC}\leq0)$',
         fontsize=12, ha='center', va='bottom', 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.text(0.5, 0.82, 
         r'$W = 1.5^{(W^{CC} + 1)/2}$',
         fontsize=12, ha='center', va='bottom', 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 设置坐标轴
plt.xlabel(r'Correlation Coefficient $V^{CC}$', fontsize=14)
plt.ylabel('Weight Value', fontsize=14)

# 设置网格和范围
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(-1.1, 1.1)
plt.ylim(-0.8, 2.0)  # 调整Y轴范围以适应新曲线

# 添加参考线
plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
plt.axvline(0, color='black', linewidth=0.8, linestyle='-')

# 显示图例
plt.legend(loc='upper left', framealpha=1)

# 保存和显示
plt.tight_layout()
plt.savefig('WCC_transformation_curves.png', bbox_inches='tight', dpi=300)
plt.show()