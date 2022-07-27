## 一、说明

Sen1Floods11-master中的代码提供了一个处理哨兵一号拍摄图像的案例，包括数据集的读取和预处理，最终将数据集构建成`torch.utils.data.DataLoader`格式，用于后面的模型训练，下面记录Sen1Floods11数据集的具体情况和一些函数的作用。

## 二、Sen1Floods11

该代码的测试数据集为S1（二通道灰度图）和Labels（单通道灰度图）,flood_test_data.csv

## 三、函数介绍

1. load_flood_test_data

   ```python
   def load_flood_train_data(input_root, label_root):
       '''
       加载训练集，返回预处理后的数据
       input_root：输入图像数据的目录
       label_root：图像对应的标签目录
       输出：初步预处理过的训练集，以列表的形式存放
       '''
       # 记录所有训练集图像名称
       fname = "flood_train_data.csv"
       training_files = []
       with open(fname) as f:
           for line in csv.reader(f):
               training_files.append(tuple((input_root+line[0], label_root+line[1])))
       return download_flood_water_data_from_list(training_files)
   ```

2. download_flood_water_data_from_list

   ```python
   def download_flood_water_data_from_list(l):
       '''
       l：元组类型，l[0]原图像路径，l[1]mask图像路径
       输出：初步预处理过的训练集，以列表的形式存放，每一项是一个元组
       '''
       i = 0
       tot_nan = 0
       tot_good = 0
       flood_data = []
       for (im_fname, mask_fname) in l:
           if not os.path.exists(os.path.join("files/", im_fname)):
               continue
           # 用0代替nan
           arr_x = np.nan_to_num(getArrFlood(os.path.join("files/", im_fname)))
           arr_y = getArrFlood(os.path.join("files/", mask_fname))
           # arr_y.max()=1,arr_y.min()=-1
           arr_y[arr_y == -1] = 255 
           # numpy.clip(a, a_min, a_max, out=None)
           # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max;
           # 小于a_min的就使得它等于a_min。
           arr_x = np.clip(arr_x, -50, 1)
           # 标准化
           arr_x = (arr_x + 50) / 51
         
       if i % 100 == 0:
           # 每处理100张图像打印一次
           print(im_fname, mask_fname)
       i += 1
       flood_data.append((arr_x,arr_y))
   
       return flood_data
   ```

3. getArrFlood

   ```python
   def getArrFlood(fname):
       '''
       利用rasterio读取tiff图像
       返回：（2，512，512）或（1，512，512）
       '''
       return rasterio.open(fname).read()
   ```

4. processAndAugment

   ```python
   def processAndAugment(data):
       # x.shape：(2, 512, 512)
       # y.shape：(1, 512, 512)
       (x,y) = data
       im,label = x.copy(), y.copy()
   
       # convert to PIL for easier transforms
       im1 = Image.fromarray(im[0])
       im2 = Image.fromarray(im[1])
       label = Image.fromarray(label.squeeze())
   
       # Get params for random transforms
       # RandomCrop从图片中随机裁剪出尺寸为size的图片
       i, j, h, w = transforms.RandomCrop.get_params(im1, (256, 256))
     
       im1 = F.crop(im1, i, j, h, w)
       im2 = F.crop(im2, i, j, h, w)
       label = F.crop(label, i, j, h, w)
       '''
       F.hflip：图像水平翻折。
   
       参数：img(PIL图像）， 要翻折的图像。
       返回：水平翻折后的图像。
       返回类型：PIL图像。
       '''
       if random.random() > 0.5:
           im1 = F.hflip(im1)
           im2 = F.hflip(im2)
           label = F.hflip(label)
       if random.random() > 0.5:
           im1 = F.vflip(im1)
           im2 = F.vflip(im2)
           label = F.vflip(label)
     
       norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
       im = torch.stack([transforms.ToTensor()(im1).squeeze(), transforms.ToTensor()(im2).squeeze()])
       im = norm(im)
       label = transforms.ToTensor()(label).squeeze()
       if torch.sum(label.gt(.003) * label.lt(.004)):
           label *= 255
       label = label.round()
   
       return im, label
   ```

