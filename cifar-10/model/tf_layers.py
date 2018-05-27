import tensorflow as tf

def get_weights(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def get_bias(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(inputs,in_filters,out_filters,kernal_size,activation):
	shape =  [kernal_size[0],kernal_size[1],in_filters,out_filters]
	W = get_weights(shape)
	b = get_bias([shape[3]])
	return activation(tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding='SAME')+b)


def max_pooling2d(inputs,pool_size,strides):
	return tf.nn.max_pool(inputs, ksize=[1, pool_size[0],pool_size[1], 1],strides=[1, strides[0],strides[0], 1], padding='SAME')


def dense(inputs,units,activation):
	in_size=int(inputs.get_shape()[1])
	W=get_weights([in_size, units])
	b=get_bias([units])
	return activation(tf.matmul(inputs,W)+b)