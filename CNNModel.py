from tensorflow.contrib.layers import flatten
import tensorflow as tf

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5,5,1,6),mean=mu,stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_W,strides=[1,1,1,1],padding='VALID') + conv1_b

    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

     # TODO: Layer 2: Convolutional. Input = 14x14x6. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5,5,6,16),mean=mu,stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1,conv2_W,strides=[1,1,1,1],padding='VALID') + conv2_b
    
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    conv2 = flatten(conv2)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(400,120),mean=mu,stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(120))
    conv3   = tf.matmul(conv2,conv3_W) + conv3_b 


    # TODO: Activation.
    conv3 = tf.nn.relu(conv3)
    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    conv4_W = tf.Variable(tf.truncated_normal(shape=(120,84),mean=mu,stddev=sigma))
    conv4_b = tf.Variable(tf.zeros(84))
    conv4   = tf.matmul(conv3,conv4_W) + conv4_b 

    # TODO: Activation.
    conv4 = tf.nn.relu(conv4)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    conv5_W = tf.Variable(tf.truncated_normal(shape=(84,10),mean=mu,stddev=sigma))
    conv5_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(conv4,conv5_W) + conv5_b
    return logits