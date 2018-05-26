from keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def LetNet(features,labels,mode):
	# Input Layer
	input_layer = features['images']

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

	predictions = {"classes":tf.argmax(input=logits,axis=1),"probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
	
	eval_metrics_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)

def main(*argv):
	letnet_classifier = tf.estimator.Estimator(model_fn=LetNet,model_dir=".cache/")
	(x_train,y_train) ,(x_test,y_test) = cifar10.load_data()
	
	y_train = y_train.astype('int32').flatten()
	y_test = y_test.astype('int32').flatten()
	
	# logging_hook = tf.train.LoggingTensorHook(every_n_iter=50)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"images": x_train.astype('float32')},y=y_train,batch_size=128,num_epochs=20,shuffle=True)

	letnet_classifier.train(input_fn=train_input_fn,)

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"images": x_test.astype('float32')},y=y_test,num_epochs=1,shuffle=False)
	eval_results = letnet_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

if __name__ == "__main__":
	tf.app.run()