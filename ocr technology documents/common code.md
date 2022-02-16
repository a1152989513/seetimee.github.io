```python
#可视化图片
%matplotlib inline
from PIL import Image
img = Image.open('./###')
img = np.array(img)

# 画出读取的图片
plt.figure(figsize=(10, 10))
plt.imshow(img)

img = Image.open('./###')
img = np.array(img)

# 画出读取的图片
plt.figure(figsize=(10, 10))
plt.imshow(img)


# 检查数据集所在路径
!tree -L 3 /home/aistudio/data


!cd /home/aistudio/data/data87746 && unzip  训练数据集.zip

# 从gitee上下载PaddleOCR代码，也可以从GitHub链接下载
!git clone https://gitee.com/paddlepaddle/PaddleOCR.git


# 检查源代码文件结构
# !cd work; mkdir model
!tree /home/aistudio/work/ -L 2



```
