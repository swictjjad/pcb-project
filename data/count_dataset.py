
# -*- coding: utf-8 -*-
"""统计 DeepPCB 数据集"""
import pathlib

p = pathlib.Path('DeepPCB-master/PCBData')

# 统计图片
test_imgs = list(p.rglob('*_test.jpg'))
temp_imgs = list(p.rglob('*_temp.jpg'))
all_imgs = test_imgs + temp_imgs

# 统计标注
labels = list(p.rglob('*_not/*.txt'))

print(f"DeepPCB 数据集统计:")
print(f"  缺陷测试图 (_test.jpg): {len(test_imgs)}")
print(f"  模板图 (_temp.jpg): {len(temp_imgs)}")
print(f"  总图片数: {len(all_imgs)}")
print(f"  标注文件数: {len(labels)}")
print(f"")
print(f"总共可用样本: ~{len(test_imgs)} 对 (缺陷图+模板图)")
