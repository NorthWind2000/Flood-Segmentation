## 一、什么是TIFF？

 TIFF是Tagged Image File Format的缩写。在现在的标准中，只有TIFF存在， 其他的提法已经舍弃不用了。做为一种标记语言，TIFF与其他文件格式最大的不同在于除了图像数据，它还可以记录很多图像的其他信息。它记录图像数据的方式也比较灵活， 理论上来说， 任何其他的图像格式都能为TIFF所用， 嵌入到TIFF里面。比如JPEG， Lossless JPEG， JPEG2000和任意数据宽度的原始无压缩数据都可以方便的嵌入到TIFF中去。由于它的可扩展性， TIFF在数字影响、遥感、医学等领域中得到了广泛的应用。TIFF文件的后缀是.tif或者.tiff。

## 二、rasterio的用法

- **读取**

数据读取操作如下：

```python
import rasterio
import os
import numpy as np
%matplotlib inline

# 数据路径，根据自己的需求设定
data_dir = "data"
fp = os.path.join(data_dir, "Bolivia_23014_S1Hand.tif")
raster = rasterio.open(fp)
type(raster)
```

数据读取之后，需要查看栅格的属性

地理栅格数据的主要的属性有坐标系、仿射变换（Affine transform）、维度、波段、缺失值格式以及边界等，实现如下

```python
raster.crs # 坐标系
raster.transform # 仿射变换
(raster.width,raster.height) # 维度
raster.count # 波段
raster.nodatavals # 缺失值
raster.driver # 数据格式

# 上面的所有信息，也可以通过raster.meta一次展示
raster.meta
```

如果需要查看波段的具体的数值以及统计信息，则可以通过如下的操作实现

```python
band1 = raster.read(1) # 第一波段属性值

# 所有波段
array = raster.read()
# 统计每个波段的信息
stats = []
for band in array:
    stats.append({
        'min': band.min(),
        'mean': band.mean(),
        'median': np.median(band),
        'max': band.max()})
# 展示统计信息
stats
```
