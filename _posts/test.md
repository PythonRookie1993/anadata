---
layout: default
title: 你好，世界
---


# Numpy
> 摘自 [《利用Python进行数据分析·第2版》](https://www.jianshu.com/p/a380222a3292)

### Numpy部分功能介绍:
* ndarray，一个具有矢量算术运算和复杂广播能力的快速且节省空间的多维数组。
* 用于对整组数据进行快速运算的标准数学函数（无需编写循环）。
* 用于读写磁盘数据的工具以及用于操作内存映射文件的工具。
线性代数、随机数生成以及傅里叶变换功能。
* 用于集成由C、C++、Fortran等语言编写的代码的A C API。

### Python Numpy对于数值计算特点:
#### 高效处理大数组的数据
而为什么呢? 这是因为:
* NumPy是在一个连续的内存块中存储数据，独立于其他Python内置对象。NumPy的C语言编写的算法库可以操作内存，而不必进行类型检查或其它前期工作。比起Python的内置序列，NumPy数组使用的内存更少。
* NumPy可以在整个数组上执行复杂的计算，而不需要Python的for循环。

### Numpy计算力展示
```Python
import numpy as np
my_arr = np.arange(1000000)
my_list = list(range(1000000))
% time for _ in range(10): my_arr2 = my_arr * 2
% time for _ in range(10): my_list2 = [x * 2 for x in my_list]
```
打印结果:
```
# Numpy计算力
CPU times: user 13 ms, sys: 12.2 ms, total: 25.3 ms   
Wall time: 26.3 ms
# python list计算
CPU times: user 359 ms, sys: 88.7 ms, total: 448 ms
Wall time: 447 ms
```

***
## 4.1 Numpy的ndarray：一种多维数组对象
NumPy最重要的一个特点就是其N维数组对象（即ndarray），该对象是一个快速而灵活的大数据集容器。你可以利用这种数组对整块数据执行一些数学运算，其语法跟标量元素之间的运算一样。

要明白Python是如何利用与标量值类似的语法进行批次计算，我先引入NumPy，然后生成一个包含随机数据的小数组：

```Python
In [12]: import numpy as np

# 生成随机数组
In [13]: data = np.random.randn(2, 3)
In [14]: data
Out[14]:
array([[-0.06031437, -1.18176615,  0.37889929],
       [-1.16309582, -0.12368684,  0.67945995]])
```
然后进行数学运算：
```python
In [15]: data * 10
Out[15]:
array([[ -0.60314374, -11.81766149,   3.78899285],
       [-11.63095822,  -1.23686845,   6.79459946]])

In [16]: data + data
Out[16]:
array([[-0.12062875, -2.3635323 ,  0.75779857],
       [-2.32619164, -0.24737369,  1.35891989]])
```
第一个例子中，所有的元素都乘以10。第二个例子中，每个元素都与自身相加。

> 笔记：在本章及全书中，我会使用标准的NumPy惯用法import numpy as np。你当然也可以在代码中使用from numpy import *，但不建议这么做。numpy的命名空间很大，包含许多函数，其中一些的名字与Python的内置函数重名（比如min和max）。

ndarray是一个通用的同构数据多维容器，也就是说，其中的所有元素必须是相同类型的。每个数组都有一个shape（一个表示各维度大小的元组）和一个dtype（一个用于说明数组数据类型的对象）：

```python
In [17]: data.shape
Out[17]: (2, 3)

In [18]: data.dtype
Out[18]: dtype('float64')
```
本章将会介绍NumPy数组的基本用法，这对于本书后面各章的理解基本够用。虽然大多数数据分析工作不需要深入理解NumPy，但是精通面向数组的编程和思维方式是成为Python科学计算牛人的一大关键步骤。
> 笔记：当你在本书中看到“数组”、“NumPy数组”、"ndarray"时，基本上都指的是同一样东西，即ndarray对象。

### 创建ndarray
创建数组最简单的办法就是使用array函数。它接受一切序列型的对象（包括其他数组），然后产生一个新的含有传入数据的NumPy数组。以一个列表的转换为例：
```python
In [19]: data1 = [6, 7.5, 8, 0, 1]

In [20]: arr1 = np.array(data1)

In [21]: arr1
Out[21]: array([ 6. ,  7.5,  8. ,  0. ,  1. ])
```
嵌套序列（比如由一组等长列表组成的列表）将会被转换为一个多维数组：
```python
In [22]: data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]

In [23]: arr2 = np.array(data2)

In [24]: arr2
Out[24]:
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])
```
因为data2是列表的列表，NumPy数组arr2的两个维度的shape是从data2引入的。可以用属性ndim和shape验证：
```python
In [25]: arr2.ndim
Out[25]: 2

In [26]: arr2.shape
Out[26]: (2, 4)
```
除非特别说明（稍后将会详细介绍），np.array会尝试为新建的这个数组推断出一个较为合适的数据类型。数据类型保存在一个特殊的dtype对象中。比如说，在上面的两个例子中，我们有：
```python
In [27]: arr1.dtype
Out[27]: dtype('float64')

In [28]: arr2.dtype
Out[28]: dtype('int64')
```
除np.array之外，还有一些函数也可以新建数组。比如，zeros和ones分别可以创建指定长度或形状的全0或全1数组。empty可以创建一个没有任何具体值的数组。要用这些方法创建多维数组，只需传入一个表示形状的元组即可：
```python
In [29]: np.zeros(10)
Out[29]: array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

In [30]: np.zeros((3, 6))
Out[30]:
array([[ 0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.]])

In [31]: np.empty((2, 3, 2))
Out[31]:
array([[[6.95187041e-310, 3.16251369e+180],
        [8.32302296e+151, 3.42926683e+169],
        [6.18057710e+223, 3.42914033e+169]],

       [[8.87707041e+252, 9.92337463e-096],
        [3.98472821e+252, 3.95970760e+257],
        [1.12856763e+277, 4.27148457e+180]]])
```
> 注意：认为np.empty会返回全0数组的想法是不安全的。很多情况下（如前所示），它返回的都是一些未初始化的垃圾值。

arange是Python内置函数range的数组版：
```Python
In [32]: np.arange(15)
Out[32]: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
```
表4-1列出了一些数组创建函数。由于NumPy关注的是数值计算，因此，如果没有特别指定，数据类型基本都是float64（浮点数）。

![表4-1 数组创建函数](http://ss1.sinaimg.cn/large/767ac7f8ly1fqfi37ufyzj20jc08xt8q&690)











作者：SeanCheney
链接：https://www.jianshu.com/p/a380222a3292
來源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
