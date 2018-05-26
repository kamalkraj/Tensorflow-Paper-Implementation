from keras.datasets import cifar10
import tensorflow as tf
from tqdm import tqdm

def get_model(weight_decay=0.0):
	
	input_layer  = tf.placeholder(tf.float32,shape=[None,32,32,3],name='input_layer')
	labels = tf.placeholder(tf.int64,shape=[None,],name='labels')

	learning_rate = tf.placeholder(tf.int32)
	# Conv Layer #1
	conv1 = tf.layers.conv2d(inputs=input_layer,filters=20,kernel_size=[5,5],padding='same',activation=tf.nn.relu)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

	# Conv Layer #2
	conv2 = tf.layers.conv2d(inputs=pool1,filters=50,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

	# Pooling Layer #2
	pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

	# Pool2 Flat #1
	pool2_flat = tf.reshape(pool2,[-1,8* 8 *50])

	# Dense Layer #1
	dense = tf.layers.dense(inputs=pool2_flat,units=128,activation=tf.nn.relu)

	# Logits Layer
	logits = tf.layers.dense(inputs=dense,units=10)

	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

	optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)

	train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())

	# classes = tf.argmax(input=logits,axis=1)
	# accuracy = tf.metrics.accuracy(labels=labels, predictions=classes)
	correct_prediction = tf.equal(tf.argmax(logits, 1),labels)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	return train_op,input_layer,labels,learning_rate,accuracy

def main():
	train_op,input_layer,labels,learning_rate,accuracy = get_model()
	(x_train,y_train) ,(x_test,y_test) = cifar10.load_data()

	y_train = y_train.astype('int64').flatten()
	y_test = y_test.astype('int64').flatten()
	epochs = 80
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		for epoch in range(epochs):
			print("epoch : ",epoch)
			start = 0
			batch_size = 128
			for _ in range(391):
				x = x_train[start:start+batch_size]
				y = y_train[start:start+batch_size]
				start += batch_size
				sess.run(train_op,feed_dict={input_layer:x,labels:y,learning_rate:0.02})
			acc = sess.run(accuracy,feed_dict={input_layer:x_test,labels:y_test,learning_rate:0.02})
			print(acc)
		# sess.run(train_op,feed_dict={input_layer:})

if __name__ == '__main__':
	main()