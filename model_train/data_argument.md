```{.python .input}
image.CreateDetAugmenter(data_shape, resize=0, rand_crop=0, rand_pad=0, rand_gray=0, rand_mirror=False, mean=None, std=None, brightness=0, contrast=0, saturation=0, pca_noise=0, hue=0, inter_method=2, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 3.0), min_eject_coverage=0.3, max_attempts=50, pad_val=(127, 127, 127))
```

提供的增强方法
0. 标准化方法：mean / std / resize  # 原因与Xavier方法与BatchNorm方法
1. HSI空间的基础图像增强技术， brightness，contrast，saturation，hue
2. PCA噪声，基础噪声图像增强技术
3. 多样性：
    rand_mirror，
    rand_gray，
    ***rand_pad，pad_val=(127, 127, 127)， rand_crop*** --尺度不变性质
4. min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 3.0), min_eject_coverage=0.3, max_attempts=50
