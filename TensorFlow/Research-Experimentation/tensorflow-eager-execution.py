#Eager Execution Basics
import tensorflow as tf

tf.enable_eager_execution()

#Tensors
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))

#Operator Overloading is also supported
print(tf.square(2) + tf.square(3))

#Each tensor has a shape and datatype
x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

#NumPy compatibility
import numpy as np
ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("And NumPy operations convert tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

#GPU Acceleration
x = tf.random_uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0: "),
print(x.device.endswith('GPU:0'))

#Datasets
#Create a source Dataset
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

#Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 1
Line2
Line3
    """)
ds_file = tf.data.TextLineDataset(filename)

#Apply Transformations
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

#Iterate
print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)

print('\nElements in ds_file:')
for x in ds_file:
    print(x)
