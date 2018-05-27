from keras.datasets import cifar10
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tf_layers import conv2d,max_pooling2d,dense

def main(unused_argv):
	def linear(x): return x
	input_layer  = tf.placeholder(tf.float32,shape=[None,32,32,3],name='input_layer')
	labels = tf.placeholder(tf.int64,shape=[None,],name='labels')

	conv1 = conv2d(input_layer,3,20,[5,5],tf.nn.relu)
	pool1 = max_pooling2d(conv1,[2,2],[2,2])
	conv2 = conv2d(pool1,20,50,[3,3],tf.nn.relu)
	pool2 = max_pooling2d(conv2,[2,2],[2,2])

	pool2_flat = tf.reshape(pool2,[-1,8* 8 *50])

	dense1 = dense(pool2_flat,128,tf.nn.relu)
	logits = dense(dense1,10,linear)

	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
	train_op = optimizer.minimize(loss=loss)#,global_step=tf.train.get_global_step())

	correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits),1),labels)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
				sess.run([train_op],feed_dict={input_layer:x,labels:y})
			acc = sess.run(accuracy,feed_dict={input_layer:x_test,labels:y_test})
			print(acc)

if __name__ == '__main__':
	tf.app.run()
