import tensorflow as tf

#initial Variable



#create input placeholder
def create_placeholder(batch_size,width,height,output_width,output_height):
    x = tf.placeholder(tf.float32,[batch_size,width,height,3])
    y = tf.placeholder(tf.float32,[batch_size,output_width,output_height,255])
    train = tf.placeholder(tf.bool)
    return x,y,train


def create_eval_placeholder(width,height):
    image = tf.placeholder(tf.float32,[1,width,height,3])
    return image

#create Leaky_Relu activation function which is not in tensorflow function library.
def leaky_relu(_input,alpha = 0.01):
    output = tf.maximum(_input,tf.multiply(_input,alpha))    
    return output

#create a convolution layer to implement convolution method
def conv2d(_input,filters,shape,stride,trainable = True):
    layer = tf.layers.conv2d(_input,filters,shape,stride,padding = 'SAME',
                             kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01))
    layer = tf.layers.batch_normalization(layer,training=trainable)
    return layer

# create a residual convolytion layer
def res_conv2d(_input,shortcut,filters,shape,stride,trainable = True):
    layer = conv2d(_input,filters,shape,training = trainable)
    residual = leaky_relu(layer+shortcut)
    return residual

#create a function to initial filter
def create_filter(size,channel_input,channel_output):
    kernel = tf.Variable(tf.truncated_normal([size,size,channel_input,channel_output]))
    return kernel

#create the main neural network
def NN(_input,trainable):
# the first group of convolutional implement
    layer = conv2d(_input,32,[3,3],trainable = trainable)
    layer = conv2d(layer,64,[3,3],(2,2),trainable = trainable)

#the start point of first residual implement
    shortcut = layer
    
#the second group of convolutional implements with residual implement
    layer = conv2d(layer,32,[1,1],trainable = trainable)
    layer = res_conv2d(layer,shortcut,128,[3,3],trainable = trainable)
    
#the third group of convolutional implements and start point of the second residual implement
    layer = conv2d(layer,128,[3,3],(2,2),trainable = trainable)
    shortcut = layer
    
#the 4th group of convolutional implements with convolution implement
    for i in range(2):
        layer = conv2d(layer,64,[1,1],trainable = trainable)
        layer = res_conv2d(layer,shortcut,128,[3,3],trainable = trainable)
#the 5th group of convolution implements and start point of the third residual implement
    layer = conv2d(layer,256,[3,3],(2,2),trainable = trainable)
    shortcut = layer
    
#the 6th group of convolutional implements with residual implements
    for i in range(8):
        layer = conv2d(layer,128,[1,1],trainable = trainable)
        layer = res_conv2d(layer,shortcut,256,[3,3],trainable = trainable)
    pre_scale3 = layer
    
#the 7th group of convolutional implement and start point of the 4th residual implement
    layer = conv2d(layer,512,[3,3],(2,2),trainable = trainable)
    shortcut = layer
    
#the 8th group of convolutional implements with residual implements
    for i in range(8):
        layer = conv2d(layer,256,[1,1],trainable = trainable)
        layer = res_conv2d(layer,shortcut,512,[3,3],trainable = trainable)
    pre_scale2 = layer
    
#the 9th group of convolutional imlement and start point of the 5th residual implement
    layer = conv2d(layer,1024,[3,3],(2,2),trainable = trainable)
    shortcut = layer
    
#the 10th group of convolutional
    for i in range(4):
        layer = conv2d(layer,512,[1,1],trainable = trainable)
        layer = res_conv2d(layer,shortcut,1024,[3,3],trainable = trainable)
    pre_scale1 = layer
    
    return pre_scale1,pre_scale2,pre_scale3
 