---
title: 'TensorFlow Fundamentals'
date: 2022-02-09
toc: true
excerpt: This post demonstrate the fundamental concepts in TensorFlow. TensorFlow is an end-to-end open souce Machine learning framework developed by Google. It can be used to implement complex deep learning models using inbuilt methods. It also blends very nicely with NumPy so we can use the Numpy Arrays in TensorFlow. This is one of the first post in the Deep learning with TensorFlow series.
permalink: /posts/2021/10/tensor-flow-fundamentals/
excerpt_separator: <!--more-->
tags:
  - cool posts
  - TensorFlow
  - Python
  - Machine learning
  - Deep Learning
  - NumPy
  - Neural Networks
---
[TensorFlow Fundamentals Notebook Machine Learning Journal Blog post](https://sandeshkatakam.github.io/My-Machine_learning-Blog/jupyter/2022/02/09/TensorFlow-Fundamentals.html)  

## Fundamental concepts of TensorFlow:  
This notebook is an account of my working for the Tensorflow tutorial by Daniel Bourke on Youtube. The Notebook covers key concepts of tensorflow essential for Deep Learning.  It also highlights key points of using the various methods of TensorFlow library and also notes the possible co|mmon errors we are going to encounter during tensorflow. The possible fixes for the errors are also included in the notebook.

**TensorFlow:**  
TensorFlow is google's open-source end to end machine learning library. The basic units of the library are tensors which are generalization of matrices to higher dimensions. TensorFlow library helps in doing the computation of tensors faster by accelerating the computation process through GPUs/TPUs.
The other important library for scientific computing is NumPy and TensorFlow works well with NumPy. The only difference is tensor flow has high functionality can be used to quickly implement the code even for complex deep learning architectures, which can help us experiment more and spend more effort on making it better rather than focussing on building the Neural Networks from scratch. You can also pass on the python functions with tensorflow to accelerate the function calls.

Concepts covered in this Notebook:
* Introduction to tensors
* Getting information from tensors
* Manipulating tensors
* Tensors and Numpy
* using @tf.function(a way to speed up your python functions)
* Using GPUs with TensorFlow (or TPUs)
* Solutions to Exercises given in the tutorial notebook.

## Introduction to Tensors


```python
# Import Tensorflow
import tensorflow as tf
print(tf.__version__)
```

    2.7.0
    


```python
# Create tensors with tf.constant()

scalar = tf.constant(7)
scalar
```




    <tf.Tensor: shape=(), dtype=int32, numpy=7>




```python
# additional examples for creating a tensor

a_scalar_1 = tf.constant(3)
a_scalar_2 = tf.constant(4)
```


```python
scalar.ndim
```




    0




```python
a_scalar_1.ndim
```




    0




```python
# Create a vector
vector = tf.constant([10,101,11])
vector
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([ 10, 101,  11], dtype=int32)>




```python
# dimensions of the above vector
vector.ndim
```




    1




```python
#create a matrix 
matrix = tf.constant([[2,3,4],[5,6,7],[8,9,0]])
matrix
```




    <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[2, 3, 4],
           [5, 6, 7],
           [8, 9, 0]], dtype=int32)>




```python
# checking the dimensions of the matrix above
matrix.ndim
```




    2




```python
# create another matrix 
another_matrix = tf.constant([[10.,7.,4.],[3.,2.,4.]], dtype =tf.float16)
another_matrix
```




    <tf.Tensor: shape=(2, 3), dtype=float16, numpy=
    array([[10.,  7.,  4.],
           [ 3.,  2.,  4.]], dtype=float16)>




```python
# create another matrix with float 32 dtypte
another_matrix_1 = tf.constant([[10.,7.,4.],[3.,2.,4.]], dtype =tf.float32)
another_matrix_1
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[10.,  7.,  4.],
           [ 3.,  2.,  4.]], dtype=float32)>



The difference between both the dtypes are precision. The higher the no after the "float" the more exact the values inside the matrix stored in your computer. 


```python
# what is the number of dimensions of the matrices we created

another_matrix.ndim
```




    2



Even though the matrix is (3,2) the ndim function gives the value 2. Because the number of elements in the shape gives us the number of dimensions of the matrix. Here, we have two elements (3,2) for the shape of the matrix so the ndim gives the output 2


```python
# another example to demonstrate ndim
example_mat = tf.constant([[[1,2,3],[3,4,5]],
                           [[6,7,3],[3,2,4]],
                           [[3,2,1],[2,1,4]]])
example_mat

```




    <tf.Tensor: shape=(3, 2, 3), dtype=int32, numpy=
    array([[[1, 2, 3],
            [3, 4, 5]],
    
           [[6, 7, 3],
            [3, 2, 4]],
    
           [[3, 2, 1],
            [2, 1, 4]]], dtype=int32)>




```python
# now let us check the number of dimensions of the matrix
example_mat.ndim
```




    3



So, we have created a matrix with shape (3,2,3) there are three elements in the value of shape. so the ndim returned the value 3


```python
#Let's create another tensor

tensor = tf.constant([[[1.,0.3,0.5],
                       [0.2,0.5,0.9],
                       [3.,6.,7.]],
                      
                      [[0.2,0.5,0.8],
                       [2.,3.5,6.7],
                       [4.,8.,0.]],
                      
                      [[2.8,5.6,7.9],
                       [0.6,7.9,6.8],
                       [3.4,5.6,7.8]]], dtype = tf.float16)
tensor
```




    <tf.Tensor: shape=(3, 3, 3), dtype=float16, numpy=
    array([[[1. , 0.3, 0.5],
            [0.2, 0.5, 0.9],
            [3. , 6. , 7. ]],
    
           [[0.2, 0.5, 0.8],
            [2. , 3.5, 6.7],
            [4. , 8. , 0. ]],
    
           [[2.8, 5.6, 7.9],
            [0.6, 7.9, 6.8],
            [3.4, 5.6, 7.8]]], dtype=float16)>




```python
# let's check the dimnesions of the tensor we created

tensor.ndim
```




    3



so, now we created a tensor of 3 dimension.

What we have created so far:

* Scalar: a single number
* Vector: a number with direction
* matrix: a two dimensional array of numbers
* Tensor: an n-dimensional array of numbers
  * 0-dimensional tensor is scalar
  * 1-dimensional tensor is vector

### Creating tensors with `tf.Variable`:


```python
# creating the same tensor with tf.Variable() as above

changeable_tensor = tf.Variable([10,7])
unchangeable_tensor = tf.constant([10,7])
changeable_tensor, unchangeable_tensor
```




    (<tf.Variable 'Variable:0' shape=(2,) dtype=int32, numpy=array([10,  7], dtype=int32)>,
     <tf.Tensor: shape=(2,), dtype=int32, numpy=array([10,  7], dtype=int32)>)




```python
# Let's try to change one of the elements in our changeable tensor
changeable_tensor[0] = 7
changeable_tensor
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-18-9972a815a90d> in <module>()
          1 # Let's try to change one of the elements in our changeable tensor
    ----> 2 changeable_tensor[0] = 7
          3 changeable_tensor
    

    TypeError: 'ResourceVariable' object does not support item assignment



```python
# Then we try .assign()

changeable_tensor[0].assign(7)
changeable_tensor
```




    <tf.Variable 'Variable:0' shape=(2,) dtype=int32, numpy=array([7, 7], dtype=int32)>




```python
# try changing the elements in unchangeable tensor
unchangeable_tensor[0] = 7

```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-20-007f4e4cfc7f> in <module>()
          1 # try changing the elements in unchangeable tensor
    ----> 2 unchangeable_tensor[0] = 7
    

    TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment



```python
unchangeable_tensor[0].assign(7)
unchangeable_tensor
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-21-958e786d8d1f> in <module>()
    ----> 1 unchangeable_tensor[0].assign(7)
          2 unchangeable_tensor
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/ops.py in __getattr__(self, name)
        440         from tensorflow.python.ops.numpy_ops import np_config
        441         np_config.enable_numpy_behavior()""".format(type(self).__name__, name))
    --> 442     self.__getattribute__(name)
        443 
        444   @staticmethod
    

    AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'assign'


As you can see the difference between `tf.Variable` and `tf.constant`. The former one is mutable and you can change and manipulate the elements using the `tf.Variable` and the latter created an immutable object where you cannot change or manipulate the values of the type `tf.Constant`

**Note:** Rarely in practice you will decide whether to use `tf.constant` or `tf.Variable` to create tensors as TensorFlow does this for you. However, if in doubt, use `tf.constant` and change it later if needed

### Creating random tensors:
Random tensors of some arbitary size which contain random numbers.
These are useful during intializing random weights at beginning of neural networks. The Neural Network then learns the paramaters using gradient descent.


```python
# create two random but the same tensors
random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape = (3,2))

another_random_1 = tf.random.Generator.from_seed(42)
another_random_1 = another_random_1.normal(shape = (3,2))

# Let's check if they are equal?

random_1, another_random_1, random_1 == another_random_1
```




    (<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
     array([[-0.7565803 , -0.06854702],
            [ 0.07595026, -1.2573844 ],
            [-0.23193763, -1.8107855 ]], dtype=float32)>,
     <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
     array([[-0.7565803 , -0.06854702],
            [ 0.07595026, -1.2573844 ],
            [-0.23193763, -1.8107855 ]], dtype=float32)>,
     <tf.Tensor: shape=(3, 2), dtype=bool, numpy=
     array([[ True,  True],
            [ True,  True],
            [ True,  True]])>)



So, the random tensors appear as random but they are infact pseudo random numbers. The seed acts like a starting trigger for the underlying random algorithm. Specifying the seed value will help us in producing the same results since the a random generator function produce the same random value everytime if we use a same seed value.

This can help when we are reproducing the same model from any where. The paramters that neural network is learning in each step will be different if we get different intialization values of our weights. If we used the same seed value as mentioned in the previously implemented model we can generate the same intialization at the beginning and produce the exact same results.


```python
random_2 = tf.random.Generator.from_seed(42) # seed is set for reproducing the same result
random_2 = random_2.normal(shape = (3,4))
```


```python
random_2
```




    <tf.Tensor: shape=(3, 4), dtype=float32, numpy=
    array([[-0.7565803 , -0.06854702,  0.07595026, -1.2573844 ],
           [-0.23193763, -1.8107855 ,  0.09988727, -0.50998646],
           [-0.7535805 , -0.57166284,  0.1480774 , -0.23362993]],
          dtype=float32)>




```python
random_3 = tf.random.Generator.from_seed(42)
random_3 = random_3.normal(shape = (4,4))
random_3
```




    <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
    array([[-0.7565803 , -0.06854702,  0.07595026, -1.2573844 ],
           [-0.23193763, -1.8107855 ,  0.09988727, -0.50998646],
           [-0.7535805 , -0.57166284,  0.1480774 , -0.23362993],
           [-0.3522796 ,  0.40621263, -1.0523509 ,  1.2054597 ]],
          dtype=float32)>




```python
# let's create a much bigger tensor
random_4 = tf.random.Generator.from_seed(21)
random_4 = random_4.normal(shape = (10,10))
random_4
```




    <tf.Tensor: shape=(10, 10), dtype=float32, numpy=
    array([[-1.322665  , -0.02279496, -0.1383193 ,  0.44207528, -0.7531523 ,
             2.0261486 , -0.06997604,  0.85445154,  0.1175475 ,  0.03493892],
           [-1.5700307 ,  0.4457582 ,  0.10944034, -0.8035768 , -1.7166729 ,
             0.3738578 , -0.14371012, -0.34646833,  1.1456194 , -0.416     ],
           [ 0.43369916,  1.0241015 , -0.74785167, -0.59090924, -1.2060374 ,
             0.8307429 ,  1.0951619 ,  1.3672234 , -0.54532146,  1.9302735 ],
           [-0.3151453 , -0.8761205 , -2.7316678 , -0.15730922,  1.3692921 ,
            -0.4367834 ,  0.8357487 ,  0.20849545,  1.4040174 , -2.735283  ],
           [ 1.2232229 , -1.8653691 ,  0.00511209, -1.0493753 ,  0.7901182 ,
             1.585549  ,  0.4356279 ,  0.23645182, -0.1589871 ,  1.302304  ],
           [ 0.9592239 ,  0.85874265, -1.5181769 ,  1.4020647 ,  1.5570306 ,
            -0.96762174,  0.495291  , -0.648484  , -1.8700892 ,  2.7830641 ],
           [-0.645002  ,  0.18022095, -0.14656258,  0.34374258,  0.41367555,
             0.17573498, -1.0871261 ,  0.45905176,  0.20386009,  0.562024  ],
           [-2.3001142 , -1.349454  ,  0.81485   ,  1.2790666 ,  0.02203509,
             1.5428121 ,  0.78953624,  0.53897345, -0.48535708,  0.74055266],
           [ 0.31662667, -1.4391748 ,  0.58923835, -1.4268045 , -0.7565803 ,
            -0.06854702,  0.07595026, -1.2573844 , -0.23193763, -1.8107855 ],
           [ 0.09988727, -0.50998646, -0.7535805 , -0.57166284,  0.1480774 ,
            -0.23362993, -0.3522796 ,  0.40621263, -1.0523509 ,  1.2054597 ]],
          dtype=float32)>




```python
# let's create a random tensor with more dimensions
random_5 = tf.random.Generator.from_seed(5)
random_5 = random_5.normal(shape = (3,3,3))
random_5
```




    <tf.Tensor: shape=(3, 3, 3), dtype=float32, numpy=
    array([[[ 1.0278524 ,  0.27974114, -0.01347923],
            [ 1.845181  ,  0.97061104, -1.0242516 ],
            [-0.6544423 , -0.29738766, -1.3240396 ]],
    
           [[ 0.28785667, -0.8757901 , -0.08857018],
            [ 0.69211644,  0.84215707, -0.06378496],
            [ 0.92800784, -0.6039789 , -0.1766927 ]],
    
           [[ 0.04221033,  0.29037967, -0.29604465],
            [-0.21134205,  0.01063002,  1.5165398 ],
            [ 0.27305737, -0.29925638, -0.3652325 ]]], dtype=float32)>




```python
# let's create another random tensor with more dimensions
random_6 = tf.random.Generator.from_seed(6)
random_6 = random_6.normal(shape = (5,5,5))
random_6
```




    <tf.Tensor: shape=(5, 5, 5), dtype=float32, numpy=
    array([[[ 0.97061104, -1.0242516 , -0.6544423 , -0.29738766,
             -1.3240396 ],
            [ 0.28785667, -0.8757901 , -0.08857018,  0.69211644,
              0.84215707],
            [-0.06378496,  0.92800784, -0.6039789 , -0.1766927 ,
              0.04221033],
            [ 0.29037967, -0.29604465, -0.21134205,  0.01063002,
              1.5165398 ],
            [ 0.27305737, -0.29925638, -0.3652325 ,  0.61883307,
             -1.0130816 ]],
    
           [[ 0.28291714,  1.2132233 ,  0.46988967,  0.37944323,
             -0.6664026 ],
            [ 0.6054596 ,  0.19181173,  0.8045827 ,  0.4769051 ,
             -0.7812124 ],
            [-0.996891  ,  0.33149973, -0.5445254 ,  1.5222508 ,
              0.59303206],
            [-0.63509274,  0.3703566 , -1.0939722 , -0.4601445 ,
              1.5420506 ],
            [-0.16822556, -0.4390865 , -0.4129243 ,  0.35877243,
             -1.9095894 ]],
    
           [[-0.2094769 ,  0.8286217 , -0.06695071, -0.35105535,
              1.0884082 ],
            [-1.3863064 ,  0.88051325, -1.6833194 ,  0.86754173,
             -0.19625713],
            [-1.322665  , -0.02279496, -0.1383193 ,  0.44207528,
             -0.7531523 ],
            [ 2.0261486 , -0.06997604,  0.85445154,  0.1175475 ,
              0.03493892],
            [-1.5700307 ,  0.4457582 ,  0.10944034, -0.8035768 ,
             -1.7166729 ]],
    
           [[ 0.3738578 , -0.14371012, -0.34646833,  1.1456194 ,
             -0.416     ],
            [ 0.43369916,  1.0241015 , -0.74785167, -0.59090924,
             -1.2060374 ],
            [ 0.8307429 ,  1.0951619 ,  1.3672234 , -0.54532146,
              1.9302735 ],
            [-0.3151453 , -0.8761205 , -2.7316678 , -0.15730922,
              1.3692921 ],
            [-0.4367834 ,  0.8357487 ,  0.20849545,  1.4040174 ,
             -2.735283  ]],
    
           [[ 1.2232229 , -1.8653691 ,  0.00511209, -1.0493753 ,
              0.7901182 ],
            [ 1.585549  ,  0.4356279 ,  0.23645182, -0.1589871 ,
              1.302304  ],
            [ 0.9592239 ,  0.85874265, -1.5181769 ,  1.4020647 ,
              1.5570306 ],
            [-0.96762174,  0.495291  , -0.648484  , -1.8700892 ,
              2.7830641 ],
            [-0.645002  ,  0.18022095, -0.14656258,  0.34374258,
              0.41367555]]], dtype=float32)>



### shuffle the order of elements in a tensor






```python
# shuffle a tensor (valuable for when you want to shuffle  )
not_shuffled = tf.constant([[10,7],
                            [3,4],
                            [2,3]])
# shuffle our non-shuffled tensor:
tf.random.shuffle(not_shuffled)
```




    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
    array([[ 3,  4],
           [ 2,  3],
           [10,  7]], dtype=int32)>




```python
not_shuffled
```




    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
    array([[10,  7],
           [ 3,  4],
           [ 2,  3]], dtype=int32)>




```python
# Let's shuffle our non-shuffled tensor again:
tf.random.shuffle(not_shuffled)
```




    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
    array([[ 3,  4],
           [10,  7],
           [ 2,  3]], dtype=int32)>




```python
# Let's shuffle our non-shuffled tensor again:
tf.random.shuffle(not_shuffled, seed = 42)
```




    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
    array([[ 2,  3],
           [ 3,  4],
           [10,  7]], dtype=int32)>




```python
# Let's shuffle our non-shuffled tensor again:
tf.random.shuffle(not_shuffled, seed = 42)
# this kind of setting the seed only work at operation-level 
# we need to declare a global seed to make this work
```




    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
    array([[ 2,  3],
           [ 3,  4],
           [10,  7]], dtype=int32)>



Even though we set the same seed the value is getting changed. Why is this happening?
refer to this link :[`tf.random.seed_set` documentation](https://www.tensorflow.org/api_docs/python/tf/random/set_seed)


```python
# Let's shuffle our non-shuffled tensor :
# Here we set the seed as global seed 
tf.random.set_seed(42)

tf.random.shuffle(not_shuffled)
```




    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
    array([[ 3,  4],
           [ 2,  3],
           [10,  7]], dtype=int32)>



#### Exercise working:
Exercise:  Read through tensorflow docs on random seed generation. Practice 5 random seed generation examples.



### Other ways to make tensors


```python
# tf.ones examples:
# we need to pass the arguments : shape, dtype etc.
# create a tensor of all ones
tf.ones([10,7])
```




    <tf.Tensor: shape=(10, 7), dtype=float32, numpy=
    array([[1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.]], dtype=float32)>




```python
# create tensor of all zeros given a certain shape
tf.zeros([10,7])
```




    <tf.Tensor: shape=(10, 7), dtype=float32, numpy=
    array([[0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>



### Turn Numpy arrays into tensors:

The main difference between Numpy and TensorFlow tensors is that tensors can be run on GPUs/TPUs.


```python
# You can also turn NumPy arrays into tensors
import numpy as np
numpy_A = np.arange(1,25,dtype = np.int32)
numpy_A
# X = tf.constant(some_matrix) # capital for tensor or matrix
# y = tf.constant(vector) # non-capital for vector
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20, 21, 22, 23, 24], dtype=int32)




```python
A = tf.constant(numpy_A, shape = (2,3,4))
B = tf.constant(numpy_A)

A,B

```




    (<tf.Tensor: shape=(2, 3, 4), dtype=int32, numpy=
     array([[[ 1,  2,  3,  4],
             [ 5,  6,  7,  8],
             [ 9, 10, 11, 12]],
     
            [[13, 14, 15, 16],
             [17, 18, 19, 20],
             [21, 22, 23, 24]]], dtype=int32)>,
     <tf.Tensor: shape=(24,), dtype=int32, numpy=
     array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24], dtype=int32)>)




```python
A.ndim
```




    3



The unmodified shape is the same shape as our NumPy vector. If you want to change the shape of the array with tf.constant, we need to make sure the product of the three values of dimensions should be equal to the no. of values in the unmodified array

So, anything we have in NumPy we can pass it to a tensor.


```python
## practice examples:
numpy_C = np.arange(1,101,dtype = np.float16)
numpy_D = np.arange(1,37,dtype = np.float32)
numpy_C, numpy_D
```




    (array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,
             12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,  22.,
             23.,  24.,  25.,  26.,  27.,  28.,  29.,  30.,  31.,  32.,  33.,
             34.,  35.,  36.,  37.,  38.,  39.,  40.,  41.,  42.,  43.,  44.,
             45.,  46.,  47.,  48.,  49.,  50.,  51.,  52.,  53.,  54.,  55.,
             56.,  57.,  58.,  59.,  60.,  61.,  62.,  63.,  64.,  65.,  66.,
             67.,  68.,  69.,  70.,  71.,  72.,  73.,  74.,  75.,  76.,  77.,
             78.,  79.,  80.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,  88.,
             89.,  90.,  91.,  92.,  93.,  94.,  95.,  96.,  97.,  98.,  99.,
            100.], dtype=float16),
     array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
            14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.,
            27., 28., 29., 30., 31., 32., 33., 34., 35., 36.], dtype=float32))




```python
# converting the NumPy arrays to tensors:
C = tf.constant(numpy_C, shape = (10,10))
D = tf.constant(numpy_D,shape = (6,6))
C,D
```




    (<tf.Tensor: shape=(10, 10), dtype=float16, numpy=
     array([[  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.],
            [ 11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.],
            [ 21.,  22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.,  30.],
            [ 31.,  32.,  33.,  34.,  35.,  36.,  37.,  38.,  39.,  40.],
            [ 41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.,  49.,  50.],
            [ 51.,  52.,  53.,  54.,  55.,  56.,  57.,  58.,  59.,  60.],
            [ 61.,  62.,  63.,  64.,  65.,  66.,  67.,  68.,  69.,  70.],
            [ 71.,  72.,  73.,  74.,  75.,  76.,  77.,  78.,  79.,  80.],
            [ 81.,  82.,  83.,  84.,  85.,  86.,  87.,  88.,  89.,  90.],
            [ 91.,  92.,  93.,  94.,  95.,  96.,  97.,  98.,  99., 100.]],
           dtype=float16)>, <tf.Tensor: shape=(6, 6), dtype=float32, numpy=
     array([[ 1.,  2.,  3.,  4.,  5.,  6.],
            [ 7.,  8.,  9., 10., 11., 12.],
            [13., 14., 15., 16., 17., 18.],
            [19., 20., 21., 22., 23., 24.],
            [25., 26., 27., 28., 29., 30.],
            [31., 32., 33., 34., 35., 36.]], dtype=float32)>)




```python
# dimensions of the C and D tensors created:
C.ndim, D.ndim
```




    (2, 2)



### Getting Information from tensors:
Attributes:  
When dealing with tensors you probably want to e aware of the following attributes:
* Shape : The length of each of the dimensions of a tensor
  * code: `tensor.shape`
* Rank: The number of tensor dimensions. A scalar has a rank 0, vector has rank 1,a matrix is rank 2, a tensor has a rank n.
  * code: `tensor.ndim`
* Axis or dimension : A particular dimension of a tensor
  * code: `tensor[0]`, `tensor[:,1]` etc
* Size : The total number of items in the tensor.
  * code: tf.size(tensor)


```python
# create a rank 4 tensor(4 dimensions)
rank_4_tensor = tf.zeros(shape = [2,3,4,5])
rank_4_tensor
```




    <tf.Tensor: shape=(2, 3, 4, 5), dtype=float32, numpy=
    array([[[[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]],
    
            [[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]],
    
            [[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]]],
    
    
           [[[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]],
    
            [[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]],
    
            [[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]]]], dtype=float32)>




```python
rank_4_tensor[0]
```




    <tf.Tensor: shape=(3, 4, 5), dtype=float32, numpy=
    array([[[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]],
    
           [[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]],
    
           [[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]]], dtype=float32)>




```python
# get the 
rank_4_tensor[:,1]
```




    <tf.Tensor: shape=(2, 4, 5), dtype=float32, numpy=
    array([[[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]],
    
           [[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]]], dtype=float32)>




```python
rank_4_tensor.shape, rank_4_tensor.ndim, tf.size(rank_4_tensor)
```




    (TensorShape([2, 3, 4, 5]), 4, <tf.Tensor: shape=(), dtype=int32, numpy=120>)




```python
# get various attributes of tensor:
print("Datatype of every element", rank_4_tensor.dtype)
print("Number of dimensions (rank): ", rank_4_tensor.ndim)
print("Shape of tensor: ", rank_4_tensor.shape)
print("Elements along the 0 axis:", rank_4_tensor.shape[0])
print("Elements along the last axis:", rank_4_tensor.shape[-1])
print("Total number of elements in our tensor:",tf.size(rank_4_tensor).numpy() )
```

    Datatype of every element <dtype: 'float32'>
    Number of dimensions (rank):  4
    Shape of tensor:  (2, 3, 4, 5)
    Elements along the 0 axis: 2
    Elements along the last axis: 5
    Total number of elements in our tensor: 120
    


```python
# sample practice
# we can put all the print statements in a function to reuse it whenever we want
def print_attributes_of_tensor(tensor_name):
   print("Datatype of every element", tensor_name.dtype)
   print("Number of dimensions (rank): ", tensor_name.ndim)
   print("Shape of tensor: ", tensor_name.shape)
   print("Elements along the 0 axis:", tensor_name.shape[0])
   print("Elements along the last axis:", tensor_name.shape[-1])
   print("Total number of elements in our tensor:",tf.size(tensor_name).numpy())
   return 0

```


```python
print_attributes_of_tensor(rank_4_tensor)
```

    Datatype of every element <dtype: 'float32'>
    Number of dimensions (rank):  4
    Shape of tensor:  (2, 3, 4, 5)
    Elements along the 0 axis: 2
    Elements along the last axis: 5
    Total number of elements in our tensor: 120
    




    0



Now, we can reuse the function to print the attributes of any tensor by passing the tensor name as function argument. We can add more print statement to display more attributes of the tensor.

### Indexing tensors:
Tensors can be indexed just like Python Lists


```python
some_list = [1,2,3,4]
some_list[:2]
```




    [1, 2]




```python
some_list[:1]
```




    [1]




```python
# get the first 2 elements along each dimension
rank_4_tensor[:2,:2,:2,:2]
```




    <tf.Tensor: shape=(2, 2, 2, 2), dtype=float32, numpy=
    array([[[[0., 0.],
             [0., 0.]],
    
            [[0., 0.],
             [0., 0.]]],
    
    
           [[[0., 0.],
             [0., 0.]],
    
            [[0., 0.],
             [0., 0.]]]], dtype=float32)>




```python
rank_4_tensor.shape
```




    TensorShape([2, 3, 4, 5])




```python
# Get the first element from each dimension from each index except for the final one
rank_4_tensor[:,:1,:1,:]
```




    <tf.Tensor: shape=(2, 1, 1, 5), dtype=float32, numpy=
    array([[[[0., 0., 0., 0., 0.]]],
    
    
           [[[0., 0., 0., 0., 0.]]]], dtype=float32)>




```python
# create a rank 2 tensor(2 dimensions)
rank_2_tensor = tf.constant([[10,7],
                             [3,4]])
rank_2_tensor
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[10,  7],
           [ 3,  4]], dtype=int32)>




```python
print_attributes_of_tensor(rank_2_tensor)
```

    Datatype of every element <dtype: 'int32'>
    Number of dimensions (rank):  2
    Shape of tensor:  (2, 2)
    Elements along the 0 axis: 2
    Elements along the last axis: 2
    Total number of elements in our tensor: 4
    




    0




```python
some_list, some_list[-1]
```




    ([1, 2, 3, 4], 4)




```python
# Get the last item of each of row of our rank2 tensor:
rank_2_tensor[:,-1]
```




    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([7, 4], dtype=int32)>




```python
# Add in extra dimension to our rank 2 tensor
rank_3_tensor = rank_2_tensor[..., tf.newaxis]
rank_3_tensor
```




    <tf.Tensor: shape=(2, 2, 1), dtype=int32, numpy=
    array([[[10],
            [ 7]],
    
           [[ 3],
            [ 4]]], dtype=int32)>



**we added a new dimension at the end**  
"..." means indicating all the other previous present dimensions and the new axis gets added at the end


```python
# Alternative to tf.newaxis
tf.expand_dims(rank_2_tensor, axis = -1) # "-1" means expand the final axis
# see the documentation for more details
```




    <tf.Tensor: shape=(2, 2, 1), dtype=int32, numpy=
    array([[[10],
            [ 7]],
    
           [[ 3],
            [ 4]]], dtype=int32)>




```python
# Expand the 0-axis:
tf.expand_dims(rank_2_tensor, axis = 0) # expand the 0-axis

```




    <tf.Tensor: shape=(1, 2, 2), dtype=int32, numpy=
    array([[[10,  7],
            [ 3,  4]]], dtype=int32)>




```python
# one more example:
tf.expand_dims(rank_2_tensor, axis = 1)
```




    <tf.Tensor: shape=(2, 1, 2), dtype=int32, numpy=
    array([[[10,  7]],
    
           [[ 3,  4]]], dtype=int32)>




```python
# More practice on indexing tensors
```

### Manipulating tensors(tensors operations):

**Basic Operations** 
`+`,`-`,`*`,`/`



```python
# You can add values to a tensor using the addition operator
tensor = tf.constant([[10,7],
                      [3,4]])
tensor+10
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[20, 17],
           [13, 14]], dtype=int32)>




```python
tensor*15
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[150, 105],
           [ 45,  60]], dtype=int32)>




```python
tensor - 10
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[ 0, -3],
           [-7, -6]], dtype=int32)>




```python
tensor /10
```




    <tf.Tensor: shape=(2, 2), dtype=float64, numpy=
    array([[1. , 0.7],
           [0.3, 0.4]])>




```python
# original tensor remains the same
tensor
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[10,  7],
           [ 3,  4]], dtype=int32)>




```python
# we can use the tensor flow built-in function also
tf.math.multiply(tensor,3)
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[30, 21],
           [ 9, 12]], dtype=int32)>




```python
# we can use alias and avoid using tf.math.add instead we can just use tf.add
tf.add(tensor,tensor)
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[20, 14],
           [ 6,  8]], dtype=int32)>



### Matrix Multiplication in TensorFlow:

In Machine learning, matrix multiplication is one of the most common tensor operations.


```python
# Matrix Multiplication in tensorflow: tf.linalg.matmul
print(tensor)
tf.linalg.matmul(tensor, tensor) # or tf.matmul also works

```

    tf.Tensor(
    [[10  7]
     [ 3  4]], shape=(2, 2), dtype=int32)
    




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[121,  98],
           [ 42,  37]], dtype=int32)>




```python
tensor, tensor
```




    (<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
     array([[10,  7],
            [ 3,  4]], dtype=int32)>, <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
     array([[10,  7],
            [ 3,  4]], dtype=int32)>)




```python
# It is performing the element-wise multiplication on tensors
tensor * tensor
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[100,  49],
           [  9,  16]], dtype=int32)>




```python
# matrix multiplication with Python opertor "@"
tensor @ tensor
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[121,  98],
           [ 42,  37]], dtype=int32)>




```python
#check the shape of the tensor
tensor.shape
```




    TensorShape([2, 2])




```python
# create a tensor (3,2) 
X = tf.constant([[1,2],
                 [3,4],
                 [5,6]])
# create another (3,2) 
Y = tf.constant([[7,8],
                 [9,10],
                 [11,12]])            
                 
```


```python
# Try to matrix multiply tensors of same shape.
X @ Y
''' This gives an error because X and Y doesn't satisfy 
the criteria for matrix multiplication '''
```


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    <ipython-input-77-04133f28c872> in <module>()
          1 # Try to matrix multiply tensors of same shape.
    ----> 2 X @ Y
          3 ''' This gives an error because X and Y doesn't satisfy 
          4 the criteria for matrix multiplication '''
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
        151     except Exception as e:
        152       filtered_tb = _process_traceback_frames(e.__traceback__)
    --> 153       raise e.with_traceback(filtered_tb) from None
        154     finally:
        155       del filtered_tb
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
       7105 def raise_from_not_ok_status(e, name):
       7106   e.message += (" name: " + name if name is not None else "")
    -> 7107   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
       7108 
       7109 
    

    InvalidArgumentError: Matrix size-incompatible: In[0]: [3,2], In[1]: [3,2] [Op:MatMul]



```python
# Try to matrix multiply tensors of same shape.
tf.matmul(X,Y)
''' This gives an error because X and Y doesn't satisfy 
the criteria for matrix multiplication '''
```


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    <ipython-input-78-6bfc63f024cf> in <module>()
          1 # Try to matrix multiply tensors of same shape.
    ----> 2 tf.matmul(X,Y)
          3 ''' This gives an error because X and Y doesn't satisfy 
          4 the criteria for matrix multiplication '''
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
        151     except Exception as e:
        152       filtered_tb = _process_traceback_frames(e.__traceback__)
    --> 153       raise e.with_traceback(filtered_tb) from None
        154     finally:
        155       del filtered_tb
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
       7105 def raise_from_not_ok_status(e, name):
       7106   e.message += (" name: " + name if name is not None else "")
    -> 7107   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
       7108 
       7109 
    

    InvalidArgumentError: Matrix size-incompatible: In[0]: [3,2], In[1]: [3,2] [Op:MatMul]


This fails because for two matrices to be multiplied the dimensions should satisfy these two  criteria: 
* Inner dimensions must match
* The resulting matrix has the shape of the inner dimensions


```python
# Make the inner dimensions of X and Y match and then mutliply the matrices
tf.reshape(Y, shape = (2,3))

```




    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[ 7,  8,  9],
           [10, 11, 12]], dtype=int32)>




```python
# now we mutliply and check
tf.matmul(X,Y)

```


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    <ipython-input-80-9e9781d51065> in <module>()
          1 # now we mutliply and check
    ----> 2 tf.matmul(X,Y)
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
        151     except Exception as e:
        152       filtered_tb = _process_traceback_frames(e.__traceback__)
    --> 153       raise e.with_traceback(filtered_tb) from None
        154     finally:
        155       del filtered_tb
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
       7105 def raise_from_not_ok_status(e, name):
       7106   e.message += (" name: " + name if name is not None else "")
    -> 7107   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
       7108 
       7109 
    

    InvalidArgumentError: Matrix size-incompatible: In[0]: [3,2], In[1]: [3,2] [Op:MatMul]



```python
# we can write all together at once like this
tf.matmul(X, tf.reshape(Y, shape = (2,3)))
```




    <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[ 27,  30,  33],
           [ 61,  68,  75],
           [ 95, 106, 117]], dtype=int32)>



This works !!!


```python
# Now we try reshaping the X instead of Y
tf.matmul(tf.reshape(X, shape = (2,3)), Y)
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[ 58,  64],
           [139, 154]], dtype=int32)>




```python
X.shape, tf.reshape(Y, shape = (2,3))
```




    (TensorShape([3, 2]), <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
     array([[ 7,  8,  9],
            [10, 11, 12]], dtype=int32)>)



You can see that the inner dimensions now match, and the output of the dot product is the same as outer 

**Note:** Matrix Multiplication is also called the "Dot Product"


```python
# can do the same thing with transpose
X, tf.transpose(X), tf.reshape(X, shape= (3,2))

```




    (<tf.Tensor: shape=(3, 2), dtype=int32, numpy=
     array([[1, 2],
            [3, 4],
            [5, 6]], dtype=int32)>, <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
     array([[1, 3, 5],
            [2, 4, 6]], dtype=int32)>, <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
     array([[1, 2],
            [3, 4],
            [5, 6]], dtype=int32)>)



#### The dot product
You can perform matrix multiplication using:
* `tf.matmul()`
* `tf.tensordot()`
* `@`

Perform the dot product on X and Y (requires X or Y to be transposed)


```python
tf.tensordot(tf.transpose(X), Y , axes =1 )
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[ 89,  98],
           [116, 128]], dtype=int32)>



 we can use either transpose or reshape


```python
# Perform matrix multiplication between X and Y
tf.matmul(X, tf.transpose(Y))
```




    <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[ 23,  29,  35],
           [ 53,  67,  81],
           [ 83, 105, 127]], dtype=int32)>




```python
# perform matrix multiplication between X and Y
tf.matmul(X, tf.reshape(Y, shape = (2,3)))
```




    <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[ 27,  30,  33],
           [ 61,  68,  75],
           [ 95, 106, 117]], dtype=int32)>



we are getting different values for the dot product in the above two cases. That means `tf.reshape()` and `tf.transpose()` does not exactly do the same thing. In some cases we might get output of both the functions same but not always.


```python
# Check values of Y, reshape Y and transposed Y
print("Normal Y:")
print(Y, "\n")

print("Y reshaped to (2,3):")
print(tf.reshape(Y, (2,3)),"\n")

print("Y transposed:")
print(tf.transpose(Y))

```

    Normal Y:
    tf.Tensor(
    [[ 7  8]
     [ 9 10]
     [11 12]], shape=(3, 2), dtype=int32) 
    
    Y reshaped to (2,3):
    tf.Tensor(
    [[ 7  8  9]
     [10 11 12]], shape=(2, 3), dtype=int32) 
    
    Y transposed:
    tf.Tensor(
    [[ 7  9 11]
     [ 8 10 12]], shape=(2, 3), dtype=int32)
    


```python
tf.matmul(X, tf.transpose(Y))
```




    <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[ 23,  29,  35],
           [ 53,  67,  81],
           [ 83, 105, 127]], dtype=int32)>



Generally, when performing matrix multiplication on two tensors, and one of the axes doesn't line up, you will transpose rather than reshape one of the tensors to satisfy the matrix multiplication rules.

### Changing the datatype of a tensor

The default datatype for tensors is int32 but however if you want to use other datatype for your tensor. we can change the datatype of the tensor.


```python
# Create a new tensor with default datatype
B = tf.constant([1.7,7.4])
B.dtype
```




    tf.float32




```python
C = tf.constant([7,10])
C.dtype
```




    tf.int32




```python
# Change from float32 to float16 -- This is reduced precision
D = tf.cast(B, dtype = tf.float16)
D, D.dtype
```




    (<tf.Tensor: shape=(2,), dtype=float16, numpy=array([1.7, 7.4], dtype=float16)>,
     tf.float16)




```python
# change from int32 to float32
E = tf.cast(C, dtype = tf.float32)
E.dtype
```




    tf.float32




```python
E_float16 = tf.cast(E, dtype = tf.float16)
E_float16
```




    <tf.Tensor: shape=(2,), dtype=float16, numpy=array([ 7., 10.], dtype=float16)>



### Aggregating tensors

Aggregating tensors  = condensing them from multiple values down to a smaller amount of values.


```python
# Get a aboslute values 
D = tf.constant([-7,10])
D
```




    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([-7, 10], dtype=int32)>




```python
# Get the absolute values
tf.abs(D)
```




    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([ 7, 10], dtype=int32)>



Let's go through the following forms of aggregation:
* Get the minimum
* Get the maximum
* Get the mean of a tensor
* Get the sum of a tensor


```python
# create a random tensor with values between 0 and 100 of size 50
E = tf.constant(np.random.randint(0,100,size = 50))
E
```




    <tf.Tensor: shape=(50,), dtype=int64, numpy=
    array([50, 30,  8,  9, 11, 58, 26, 64, 89,  4, 96, 20, 19, 35, 25, 33, 53,
           45, 15, 73, 59, 29, 22, 16, 54, 65, 16, 87, 25, 13, 50, 35, 77, 10,
           88, 34, 49, 70, 99, 37,  3, 93, 98, 48, 50, 35, 66, 97, 37, 93])>




```python
tf.size(E), E.shape, E.ndim
```




    (<tf.Tensor: shape=(), dtype=int32, numpy=50>, TensorShape([50]), 1)




```python
# Find the minimum 
tf.reduce_min(E)
```




    <tf.Tensor: shape=(), dtype=int64, numpy=3>




```python
# Find the maximum 
tf.reduce_max(E)
```




    <tf.Tensor: shape=(), dtype=int64, numpy=99>




```python
# Find the mean
tf.reduce_mean(E)
```




    <tf.Tensor: shape=(), dtype=int64, numpy=46>




```python
# Find the sum
tf.reduce_sum(E)
```




    <tf.Tensor: shape=(), dtype=int64, numpy=2318>



**Exercise:**   
Find the variance and standard deviation of our `E` tensor using TensorFlow methods


```python
# To find the variance of the tensor we need to access tfp module
# Find the variance of our tensor
import tensorflow_probability as tfp
tfp.stats.variance(E)
```




    <tf.Tensor: shape=(), dtype=int64, numpy=832>




```python
tf.math.reduce_std(E)
# Error : The input must be either real or complex
# so cast it to float32 
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-120-a76c926197b7> in <module>()
    ----> 1 tf.math.reduce_std(E)
          2 # Error : The input must be either real or complex
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
        151     except Exception as e:
        152       filtered_tb = _process_traceback_frames(e.__traceback__)
    --> 153       raise e.with_traceback(filtered_tb) from None
        154     finally:
        155       del filtered_tb
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/math_ops.py in reduce_variance(input_tensor, axis, keepdims, name)
       2673     means = reduce_mean(input_tensor, axis=axis, keepdims=True)
       2674     if means.dtype.is_integer:
    -> 2675       raise TypeError(f"Input must be either real or complex. "
       2676                       f"Received integer type {means.dtype}.")
       2677     diff = input_tensor - means
    

    TypeError: Input must be either real or complex. Received integer type <dtype: 'int64'>.



```python
# Find the standard deviation
tf.math.reduce_std(tf.cast(E, dtype = tf.float32))
# The method works only if the tensor elements are either real or complex
```




    <tf.Tensor: shape=(), dtype=float32, numpy=28.844936>



### Find the positional maximum and minimum

We find this helpful for our output probabilities that come from neural network.


```python
# Create a new tensor for finding positional minimum and maximum
tf.random.set_seed(42)
F = tf.random.uniform(shape =[50])
F
```




    <tf.Tensor: shape=(50,), dtype=float32, numpy=
    array([0.6645621 , 0.44100678, 0.3528825 , 0.46448255, 0.03366041,
           0.68467236, 0.74011743, 0.8724445 , 0.22632635, 0.22319686,
           0.3103881 , 0.7223358 , 0.13318717, 0.5480639 , 0.5746088 ,
           0.8996835 , 0.00946367, 0.5212307 , 0.6345445 , 0.1993283 ,
           0.72942245, 0.54583454, 0.10756552, 0.6767061 , 0.6602763 ,
           0.33695042, 0.60141766, 0.21062577, 0.8527372 , 0.44062173,
           0.9485276 , 0.23752594, 0.81179297, 0.5263394 , 0.494308  ,
           0.21612847, 0.8457197 , 0.8718841 , 0.3083862 , 0.6868038 ,
           0.23764038, 0.7817228 , 0.9671384 , 0.06870162, 0.79873943,
           0.66028714, 0.5871513 , 0.16461694, 0.7381023 , 0.32054043],
          dtype=float32)>




```python
# Find the positional max
tf.argmax(F)
```




    <tf.Tensor: shape=(), dtype=int64, numpy=42>




```python
np.argmax(F)
```




    42




```python
# Index on our largest value position
F[tf.argmax(F)]
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.9671384>




```python
# Check for equality
assert F[tf.argmax(F)] == tf.reduce_max(F)
```

No error so we got it right!


```python
F[tf.argmax(F)] == tf.reduce_max(F)
```




    <tf.Tensor: shape=(), dtype=bool, numpy=True>




```python
# Find the positional minimum
tf.argmin(F)
```




    <tf.Tensor: shape=(), dtype=int64, numpy=16>




```python
# Find the minimum using the positional minimum Index
F[tf.argmin(F)]
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.009463668>



### Squeezing a tensor (removing all single dimensions)


```python
# Create a tensor to get started
tf.random.set_seed(42)
G = tf.constant(tf.random.uniform(shape = [50]), shape = (1,1,1,1,50))
G
```




    <tf.Tensor: shape=(1, 1, 1, 1, 50), dtype=float32, numpy=
    array([[[[[0.6645621 , 0.44100678, 0.3528825 , 0.46448255, 0.03366041,
               0.68467236, 0.74011743, 0.8724445 , 0.22632635, 0.22319686,
               0.3103881 , 0.7223358 , 0.13318717, 0.5480639 , 0.5746088 ,
               0.8996835 , 0.00946367, 0.5212307 , 0.6345445 , 0.1993283 ,
               0.72942245, 0.54583454, 0.10756552, 0.6767061 , 0.6602763 ,
               0.33695042, 0.60141766, 0.21062577, 0.8527372 , 0.44062173,
               0.9485276 , 0.23752594, 0.81179297, 0.5263394 , 0.494308  ,
               0.21612847, 0.8457197 , 0.8718841 , 0.3083862 , 0.6868038 ,
               0.23764038, 0.7817228 , 0.9671384 , 0.06870162, 0.79873943,
               0.66028714, 0.5871513 , 0.16461694, 0.7381023 , 0.32054043]]]]],
          dtype=float32)>




```python
G.shape
```




    TensorShape([1, 1, 1, 1, 50])




```python
G_squeezed = tf.squeeze(G)
G_squeezed, G_squeezed.shape
```




    (<tf.Tensor: shape=(50,), dtype=float32, numpy=
     array([0.6645621 , 0.44100678, 0.3528825 , 0.46448255, 0.03366041,
            0.68467236, 0.74011743, 0.8724445 , 0.22632635, 0.22319686,
            0.3103881 , 0.7223358 , 0.13318717, 0.5480639 , 0.5746088 ,
            0.8996835 , 0.00946367, 0.5212307 , 0.6345445 , 0.1993283 ,
            0.72942245, 0.54583454, 0.10756552, 0.6767061 , 0.6602763 ,
            0.33695042, 0.60141766, 0.21062577, 0.8527372 , 0.44062173,
            0.9485276 , 0.23752594, 0.81179297, 0.5263394 , 0.494308  ,
            0.21612847, 0.8457197 , 0.8718841 , 0.3083862 , 0.6868038 ,
            0.23764038, 0.7817228 , 0.9671384 , 0.06870162, 0.79873943,
            0.66028714, 0.5871513 , 0.16461694, 0.7381023 , 0.32054043],
           dtype=float32)>, TensorShape([50]))



### One hot encoding tensors

What is one-hot encoding?  




```python
# Create a list of indices
some_list = [0,1,2,3] # could be red, green , blue , purple

# one hot encoding our list of indices 
tf.one_hot(some_list, depth = 4)
```




    <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)>




```python
# Specify custom values for one hot encoding
tf.one_hot(some_list, depth = 4, on_value = "Yo I love deep learning", off_value = "I also like to write")
```




    <tf.Tensor: shape=(4, 4), dtype=string, numpy=
    array([[b'Yo I love deep learning', b'I also like to write',
            b'I also like to write', b'I also like to write'],
           [b'I also like to write', b'Yo I love deep learning',
            b'I also like to write', b'I also like to write'],
           [b'I also like to write', b'I also like to write',
            b'Yo I love deep learning', b'I also like to write'],
           [b'I also like to write', b'I also like to write',
            b'I also like to write', b'Yo I love deep learning']],
          dtype=object)>



### More on math functions:

* squaring
* log
* square root


```python
# vreate a new tensor
H = tf.range(1,10)
H
```




    <tf.Tensor: shape=(9,), dtype=int32, numpy=array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)>




```python
tf.square(H)
```




    <tf.Tensor: shape=(9,), dtype=int32, numpy=array([ 1,  4,  9, 16, 25, 36, 49, 64, 81], dtype=int32)>




```python
tf.sqrt(H)
```


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    <ipython-input-142-f2dbaafeb52c> in <module>()
    ----> 1 tf.sqrt(H)
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
        151     except Exception as e:
        152       filtered_tb = _process_traceback_frames(e.__traceback__)
    --> 153       raise e.with_traceback(filtered_tb) from None
        154     finally:
        155       del filtered_tb
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
       7105 def raise_from_not_ok_status(e, name):
       7106   e.message += (" name: " + name if name is not None else "")
    -> 7107   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
       7108 
       7109 
    

    InvalidArgumentError: Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
    	; NodeDef: {{node Sqrt}}; Op<name=Sqrt; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Sqrt]


we got an error here because tensors of dtype int32 is not allowed as arguments for sqrt function. So, we cast it to different datatype


```python
# Find the square root
tf.sqrt(tf.cast(H, dtype = tf.float32))
```




    <tf.Tensor: shape=(9,), dtype=float32, numpy=
    array([0.99999994, 1.4142134 , 1.7320508 , 1.9999999 , 2.236068  ,
           2.4494896 , 2.6457512 , 2.8284268 , 3.        ], dtype=float32)>




```python
# Find the log
tf.math.log(H)
```


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    <ipython-input-144-d7e970c5bd0b> in <module>()
          1 # Find the log
    ----> 2 tf.math.log(H)
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/gen_math_ops.py in log(x, name)
       5468       return _result
       5469     except _core._NotOkStatusException as e:
    -> 5470       _ops.raise_from_not_ok_status(e, name)
       5471     except _core._FallbackException:
       5472       pass
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
       7105 def raise_from_not_ok_status(e, name):
       7106   e.message += (" name: " + name if name is not None else "")
    -> 7107   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
       7108 
       7109 
    

    InvalidArgumentError: Value for attr 'T' of int32 is not in the list of allowed values: bfloat16, half, float, double, complex64, complex128
    	; NodeDef: {{node Log}}; Op<name=Log; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:Log]


we also get the same error for this too. so cast the argument tensor to one of the allowed values.


```python
tf.math.log(tf.cast(H, dtype = tf.float32))
```




    <tf.Tensor: shape=(9,), dtype=float32, numpy=
    array([0.       , 0.6931472, 1.0986123, 1.3862944, 1.609438 , 1.7917595,
           1.9459102, 2.0794415, 2.1972246], dtype=float32)>



### Tensors and NumPy

NumPy is a package used for scientific computing. The most fundamental type in NumPy is numpy array.  
TensorFlow interacts beautifully with NumPy arrays.


```python
# Create a tensor directly from a Numpy array
J = tf.constant(np.array([3.,7.,10.]))
J
```




    <tf.Tensor: shape=(3,), dtype=float64, numpy=array([ 3.,  7., 10.])>




```python
# Convert our tensor back to a NumPy array
np.array(J), type(np.array(J))
```




    (array([ 3.,  7., 10.]), numpy.ndarray)




```python
# convert tensor J to a NumPy array
J.numpy() , type(J.numpy())
```




    (array([ 3.,  7., 10.]), numpy.ndarray)




```python
J = tf.constant([3.])
J.numpy()[0]
```




    3.0




```python
# The default types of each are slightly different
numpy_J = tf.constant(np.array([3.,7.,10.]))
tensor_J = tf.constant([3.,7.,10.])
# Check the datatypes of each
numpy_J.dtype , tensor_J.dtype
```




    (tf.float64, tf.float32)



We can see above that creating tensors directly from tensorflow will create a default dtype of float32 values but if we pass in numpy array to `tf.constant` the default dtype of created tensor is float64

### Using `@tf.function`  

In your TensorFlow adventures, you might come across Python functions which have the decorator [`@tf.function`](https://www.tensorflow.org/api_docs/python/tf/function).  

But in short, decorators modify a function in one way or another.  

In the `@tf.function` decorator case, it turns a Python function into a callable TensorFlow graph. Which is a fancy way of saying, if you've written your own Python function, and you decorate it with `@tf.function`, when you export your code (to potentially run on another device), TensorFlow will attempt to convert it into a fast(er) version of itself (by making it part of a computation graph).

For more on this, read the [Better performnace with tf.function](https://www.tensorflow.org/guide/function) guide.


```python
# Create a simple function
def function(x, y):
  return x ** 2 + y

x = tf.constant(np.arange(0, 10))
y = tf.constant(np.arange(10, 20))
function(x, y)
```




    <tf.Tensor: shape=(10,), dtype=int64, numpy=array([ 10,  12,  16,  22,  30,  40,  52,  66,  82, 100])>




```python
# Create the same function and decorate it with tf.function
@tf.function
def tf_function(x, y):
  return x ** 2 + y

tf_function(x, y)
```




    <tf.Tensor: shape=(10,), dtype=int64, numpy=array([ 10,  12,  16,  22,  30,  40,  52,  66,  82, 100])>



If you noticed no difference between the above two functions (the decorated one and the non-decorated one) you'd be right.

Much of the difference happens behind the scenes. One of the main ones being potential code speed-ups where possible.

### Using GPUs:


We've mentioned GPUs plenty of times throughout this notebook.

So how do you check if you've got one available?

You can check if you've got access to a GPU using [`tf.config.list_physical_devices()`](https://www.tensorflow.org/guide/gpu).


```python
print(tf.config.list_physical_devices('GPU'))
```

    []
    

The PC I am working from has no GPU support.


```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

    []
    

If you've got access to a GPU, the cell above should output something like:

`[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

You can also find information about your GPU using `!nvidia-smi`.

> 🔑 **Note:** If you have access to a GPU, TensorFlow will automatically use it whenever possible.

### Solutions to the Exercises given in the tutorial Notebook:

1. Create a vector, scalar, matrix and tensor with values of your choosing using `tf.constant()`.


```python
# solution:
A1 = tf.constant([3]) # scalar
A2 = tf.constant([10, 7]) # vector
A3 = tf.constant([[10,7],
            [3,4]]) # matrix
A4 = tf.constant([[[10,7,3],
             [3,4,5]],
            [[2,3,4],
             [7,8,9]],
            [[1,2,3],
             [6,7,8]]]) # tensor of dimension 3
            
A1,A2,A3,A4
```




    (<tf.Tensor: shape=(1,), dtype=int32, numpy=array([3], dtype=int32)>,
     <tf.Tensor: shape=(2,), dtype=int32, numpy=array([10,  7], dtype=int32)>,
     <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
     array([[10,  7],
            [ 3,  4]], dtype=int32)>,
     <tf.Tensor: shape=(3, 2, 3), dtype=int32, numpy=
     array([[[10,  7,  3],
             [ 3,  4,  5]],
     
            [[ 2,  3,  4],
             [ 7,  8,  9]],
     
            [[ 1,  2,  3],
             [ 6,  7,  8]]], dtype=int32)>)



2. Find the shape, rank and size of the tensors you created in 1.


```python
# For A1 -- scalar
tf.shape(A1), tf.size(A1), tf.rank(A1)
```




    (<tf.Tensor: shape=(1,), dtype=int32, numpy=array([1], dtype=int32)>,
     <tf.Tensor: shape=(), dtype=int32, numpy=1>,
     <tf.Tensor: shape=(), dtype=int32, numpy=1>)




```python
# For A2 -- vector
tf.shape(A2), tf.size(A2), tf.rank(A2)
```




    (<tf.Tensor: shape=(1,), dtype=int32, numpy=array([2], dtype=int32)>,
     <tf.Tensor: shape=(), dtype=int32, numpy=2>,
     <tf.Tensor: shape=(), dtype=int32, numpy=1>)




```python
# For A3 -- matrix
tf.shape(A3), tf.size(A3), tf.rank(A3)
```




    (<tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 2], dtype=int32)>,
     <tf.Tensor: shape=(), dtype=int32, numpy=4>,
     <tf.Tensor: shape=(), dtype=int32, numpy=2>)




```python
# For A4 -- tensor of 3 dimensions
tf.shape(A4), tf.size(A4), tf.rank(A4)
```




    (<tf.Tensor: shape=(3,), dtype=int32, numpy=array([3, 2, 3], dtype=int32)>,
     <tf.Tensor: shape=(), dtype=int32, numpy=18>,
     <tf.Tensor: shape=(), dtype=int32, numpy=3>)



3. Create two tensors containing random values between 0 and 1 with shape `[5, 300]`.


```python
tf.random.set_seed(42)
B1 = tf.random.uniform([5,300], minval = 0, maxval = 1) # it works even if we not specify the min and max val since the function arguments defaults to 0 and 1
B2 = tf.random.uniform([5,300], minval = 0, maxval = 1) 
B1,B2
```




    (<tf.Tensor: shape=(5, 300), dtype=float32, numpy=
     array([[0.6645621 , 0.44100678, 0.3528825 , ..., 0.31410468, 0.7593535 ,
             0.03699052],
            [0.532024  , 0.29129946, 0.10571766, ..., 0.54052293, 0.31425726,
             0.2200619 ],
            [0.08404207, 0.03614604, 0.97732127, ..., 0.21516645, 0.9786098 ,
             0.00726748],
            [0.7396945 , 0.6653172 , 0.0787828 , ..., 0.7117733 , 0.07013571,
             0.9409125 ],
            [0.15861344, 0.12024033, 0.27218235, ..., 0.8824879 , 0.1432488 ,
             0.44135118]], dtype=float32)>,
     <tf.Tensor: shape=(5, 300), dtype=float32, numpy=
     array([[0.68789124, 0.48447883, 0.9309944 , ..., 0.6920762 , 0.33180213,
             0.9212563 ],
            [0.27369928, 0.10631859, 0.6218617 , ..., 0.4382149 , 0.30427706,
             0.51477313],
            [0.00920248, 0.37280262, 0.8177401 , ..., 0.56786287, 0.49201214,
             0.9892651 ],
            [0.88608265, 0.08672249, 0.12160683, ..., 0.91770685, 0.72545695,
             0.8280058 ],
            [0.36690474, 0.9200133 , 0.9646884 , ..., 0.69012   , 0.7137332 ,
             0.2584542 ]], dtype=float32)>)



4. Multiply the two tensors you created in 3 using matrix multiplication.


```python
tf.matmul(B1,tf.transpose(B2))
```




    <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
    array([[80.33344 , 73.40498 , 77.15962 , 73.98368 , 80.90053 ],
           [75.146355, 68.80437 , 74.24302 , 71.841835, 75.60206 ],
           [79.7594  , 75.644554, 77.797585, 74.74873 , 80.559845],
           [75.085266, 69.06406 , 74.307755, 72.27616 , 76.05668 ],
           [85.056885, 74.266266, 78.00687 , 74.88678 , 83.13418 ]],
          dtype=float32)>




```python
tf.matmul(tf.transpose(B1), B2)
```




    <tf.Tensor: shape=(300, 300), dtype=float32, numpy=
    array([[1.317161  , 0.61993605, 1.2612379 , ..., 1.5290778 , 1.0735596 ,
            1.6227092 ],
           [1.0170685 , 0.42642498, 0.8181824 , ..., 1.1469344 , 0.8212255 ,
            1.1739546 ],
           [0.45034647, 0.8037954 , 1.4656199 , ..., 1.105671  , 0.88152766,
            1.481925  ],
           ...,
           [1.3204696 , 1.1634867 , 1.7423928 , ..., 1.8386563 , 1.5207756 ,
            1.5979093 ],
           [0.73207504, 0.90400356, 1.8493464 , ..., 1.3821819 , 0.98218614,
            1.924531  ],
           [1.0814031 , 0.5316744 , 0.7174167 , ..., 1.2942287 , 1.0804075 ,
            1.0476992 ]], dtype=float32)>



5. Multiply the two tensors you created in 3 using dot product.


```python
tf.tensordot(B1,tf.transpose(B2), axes = 1)
```




    <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
    array([[80.33344 , 73.40498 , 77.15962 , 73.98368 , 80.90053 ],
           [75.146355, 68.80437 , 74.24302 , 71.841835, 75.60206 ],
           [79.7594  , 75.644554, 77.797585, 74.74873 , 80.559845],
           [75.085266, 69.06406 , 74.307755, 72.27616 , 76.05668 ],
           [85.056885, 74.266266, 78.00687 , 74.88678 , 83.13418 ]],
          dtype=float32)>



6. Create a tensor with random values between 0 and 1 with shape `[224, 224, 3]`.


```python
tf.random.set_seed(42)
randtensor = tf.random.uniform([224,224,3])
randtensor
```




    <tf.Tensor: shape=(224, 224, 3), dtype=float32, numpy=
    array([[[0.6645621 , 0.44100678, 0.3528825 ],
            [0.46448255, 0.03366041, 0.68467236],
            [0.74011743, 0.8724445 , 0.22632635],
            ...,
            [0.42612267, 0.09686017, 0.16105258],
            [0.1487099 , 0.04513884, 0.9497483 ],
            [0.4393103 , 0.28527975, 0.96971095]],
    
           [[0.73308516, 0.5657046 , 0.33238935],
            [0.8838178 , 0.87544763, 0.56711245],
            [0.8879347 , 0.47661996, 0.42041814],
            ...,
            [0.7716515 , 0.9116473 , 0.3229897 ],
            [0.43050945, 0.83253574, 0.45549798],
            [0.29816985, 0.9639522 , 0.3316357 ]],
    
           [[0.41132426, 0.2179662 , 0.53570235],
            [0.5112119 , 0.6484759 , 0.8894886 ],
            [0.42459428, 0.20189774, 0.85781324],
            ...,
            [0.02888799, 0.3995477 , 0.11355484],
            [0.68524575, 0.04945195, 0.17778492],
            [0.97627187, 0.79811585, 0.9411576 ]],
    
           ...,
    
           [[0.9019445 , 0.27011132, 0.8090267 ],
            [0.32395256, 0.6672456 , 0.940673  ],
            [0.7166116 , 0.8860713 , 0.6777594 ],
            ...,
            [0.8318608 , 0.39227867, 0.68916583],
            [0.1599741 , 0.46428144, 0.4656595 ],
            [0.8619243 , 0.24755931, 0.33835268]],
    
           [[0.47570062, 0.09377229, 0.11811328],
            [0.0523994 , 0.38206005, 0.12188685],
            [0.2757113 , 0.44918692, 0.9179864 ],
            ...,
            [0.4974177 , 0.4562863 , 0.8261535 ],
            [0.60251105, 0.27676368, 0.258716  ],
            [0.7977431 , 0.74125385, 0.76062095]],
    
           [[0.4755299 , 0.4661665 , 0.14167643],
            [0.9103775 , 0.41117966, 0.83182037],
            [0.79765654, 0.38330686, 0.5313202 ],
            ...,
            [0.94517136, 0.17730081, 0.00362825],
            [0.6170398 , 0.9977623 , 0.8315122 ],
            [0.6683676 , 0.68716586, 0.4447713 ]]], dtype=float32)>



7. Find the min and max values of the tensor you created in 6.


```python
tf.reduce_max(randtensor)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.999998>




```python
tf.reduce_min(randtensor)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=3.5762787e-07>



8. Created a tensor with random values of shape `[1, 224, 224, 3]` then squeeze it to change the shape to `[224, 224, 3]`.


```python
tf.random.set_seed(42)
for_squeeze = tf.random.uniform([1,224,224,3])
for_squeeze
```




    <tf.Tensor: shape=(1, 224, 224, 3), dtype=float32, numpy=
    array([[[[0.6645621 , 0.44100678, 0.3528825 ],
             [0.46448255, 0.03366041, 0.68467236],
             [0.74011743, 0.8724445 , 0.22632635],
             ...,
             [0.42612267, 0.09686017, 0.16105258],
             [0.1487099 , 0.04513884, 0.9497483 ],
             [0.4393103 , 0.28527975, 0.96971095]],
    
            [[0.73308516, 0.5657046 , 0.33238935],
             [0.8838178 , 0.87544763, 0.56711245],
             [0.8879347 , 0.47661996, 0.42041814],
             ...,
             [0.7716515 , 0.9116473 , 0.3229897 ],
             [0.43050945, 0.83253574, 0.45549798],
             [0.29816985, 0.9639522 , 0.3316357 ]],
    
            [[0.41132426, 0.2179662 , 0.53570235],
             [0.5112119 , 0.6484759 , 0.8894886 ],
             [0.42459428, 0.20189774, 0.85781324],
             ...,
             [0.02888799, 0.3995477 , 0.11355484],
             [0.68524575, 0.04945195, 0.17778492],
             [0.97627187, 0.79811585, 0.9411576 ]],
    
            ...,
    
            [[0.9019445 , 0.27011132, 0.8090267 ],
             [0.32395256, 0.6672456 , 0.940673  ],
             [0.7166116 , 0.8860713 , 0.6777594 ],
             ...,
             [0.8318608 , 0.39227867, 0.68916583],
             [0.1599741 , 0.46428144, 0.4656595 ],
             [0.8619243 , 0.24755931, 0.33835268]],
    
            [[0.47570062, 0.09377229, 0.11811328],
             [0.0523994 , 0.38206005, 0.12188685],
             [0.2757113 , 0.44918692, 0.9179864 ],
             ...,
             [0.4974177 , 0.4562863 , 0.8261535 ],
             [0.60251105, 0.27676368, 0.258716  ],
             [0.7977431 , 0.74125385, 0.76062095]],
    
            [[0.4755299 , 0.4661665 , 0.14167643],
             [0.9103775 , 0.41117966, 0.83182037],
             [0.79765654, 0.38330686, 0.5313202 ],
             ...,
             [0.94517136, 0.17730081, 0.00362825],
             [0.6170398 , 0.9977623 , 0.8315122 ],
             [0.6683676 , 0.68716586, 0.4447713 ]]]], dtype=float32)>




```python
G_squeezed = tf.squeeze(for_squeeze)
G_squeezed, G_squeezed.shape
```




    (<tf.Tensor: shape=(224, 224, 3), dtype=float32, numpy=
     array([[[0.6645621 , 0.44100678, 0.3528825 ],
             [0.46448255, 0.03366041, 0.68467236],
             [0.74011743, 0.8724445 , 0.22632635],
             ...,
             [0.42612267, 0.09686017, 0.16105258],
             [0.1487099 , 0.04513884, 0.9497483 ],
             [0.4393103 , 0.28527975, 0.96971095]],
     
            [[0.73308516, 0.5657046 , 0.33238935],
             [0.8838178 , 0.87544763, 0.56711245],
             [0.8879347 , 0.47661996, 0.42041814],
             ...,
             [0.7716515 , 0.9116473 , 0.3229897 ],
             [0.43050945, 0.83253574, 0.45549798],
             [0.29816985, 0.9639522 , 0.3316357 ]],
     
            [[0.41132426, 0.2179662 , 0.53570235],
             [0.5112119 , 0.6484759 , 0.8894886 ],
             [0.42459428, 0.20189774, 0.85781324],
             ...,
             [0.02888799, 0.3995477 , 0.11355484],
             [0.68524575, 0.04945195, 0.17778492],
             [0.97627187, 0.79811585, 0.9411576 ]],
     
            ...,
     
            [[0.9019445 , 0.27011132, 0.8090267 ],
             [0.32395256, 0.6672456 , 0.940673  ],
             [0.7166116 , 0.8860713 , 0.6777594 ],
             ...,
             [0.8318608 , 0.39227867, 0.68916583],
             [0.1599741 , 0.46428144, 0.4656595 ],
             [0.8619243 , 0.24755931, 0.33835268]],
     
            [[0.47570062, 0.09377229, 0.11811328],
             [0.0523994 , 0.38206005, 0.12188685],
             [0.2757113 , 0.44918692, 0.9179864 ],
             ...,
             [0.4974177 , 0.4562863 , 0.8261535 ],
             [0.60251105, 0.27676368, 0.258716  ],
             [0.7977431 , 0.74125385, 0.76062095]],
     
            [[0.4755299 , 0.4661665 , 0.14167643],
             [0.9103775 , 0.41117966, 0.83182037],
             [0.79765654, 0.38330686, 0.5313202 ],
             ...,
             [0.94517136, 0.17730081, 0.00362825],
             [0.6170398 , 0.9977623 , 0.8315122 ],
             [0.6683676 , 0.68716586, 0.4447713 ]]], dtype=float32)>,
     TensorShape([224, 224, 3]))



9. Create a tensor with shape `[10]` using your own choice of values, then find the index which has the maximum value.


```python
tf.random.set_seed(42)
nine_ans = tf.random.uniform([10], maxval = 10,dtype = tf.int32)
nine_ans
```




    <tf.Tensor: shape=(10,), dtype=int32, numpy=array([7, 9, 1, 6, 2, 4, 3, 3, 1, 1], dtype=int32)>




```python
tf.argmax(nine_ans)
```




    <tf.Tensor: shape=(), dtype=int64, numpy=1>



10. One-hot encode the tensor you created in 9.


```python
tf.one_hot(tf.cast(nine_ans,dtype = tf.int32), depth = 10)
```




    <tf.Tensor: shape=(10, 10), dtype=float32, numpy=
    array([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>



**Bibliography**:
* [Tensorflow Daniel Bourke 2021 Youtube](https://youtu.be/tpCFfeUEGs8)
