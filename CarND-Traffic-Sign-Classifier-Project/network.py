import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def weight_variable(shape, mean=0, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, mean = mean, stddev = stddev))

def bias_variable(shape, val=0.):
    return tf.Variable(tf.constant(val, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
def LeNet5(x):
    mu = 0
    sigma = 0.1  # 0.01

    # 1st stage
    print('----- Stage # 1 -----')
    print('{} => {}'.format('INPUT', x))
    conv1_W = weight_variable((5, 5, 1, 32), mean=mu, stddev=sigma)
    conv1_b = bias_variable([32])
    conv1 = conv2d(x, conv1_W) + conv1_b
    conv1 = tf.nn.relu(conv1) # Input = 32x32x1. Output = 28x28x32.
    print('{} => {}'.format('RELU 1', conv1))
    conv1 = max_pool_2x2(conv1)                    # Input = 28x28x32. Output = 14x14x32.
    print('{} => {}'.format('MAX POOL 1', conv1))

    # 2nd stage:
    print('----- Stage # 2 -----')
    conv2_W = weight_variable((5, 5, 32, 64), mean=mu, stddev=sigma)
    conv2_b = bias_variable([64])
    conv2 = conv2d(conv1, conv2_W) + conv2_b       # Input = 14x14x32. Output = 14x14x64.
    conv2 = tf.nn.relu(conv2)
    print('{} => {}'.format('RELU 2', conv2))
    conv2 = max_pool_2x2(conv2)                    # Input = 14x14x64. Output = 5*5*64.
    print('{} => {}'.format('MAX POOL 2', conv2))


    print('Shape: {}'.format(conv2.get_shape().as_list()[-1]))
    # Fully connected
    # FC 1
    print('----- FC -----')
    fc0   = flatten(conv2)                        # Input = 5*5*64. Output = 5*5*64 => 1600
    print('{} => {}'.format('Flatten', fc0))
    fc1_W = weight_variable((5*5*64, 1024),  mean=mu, stddev=sigma)
    fc1_b = bias_variable([1024])
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # FC 2
    fc2_W = weight_variable((1024, 1024),  mean=mu, stddev=sigma)
    fc2_b = bias_variable([1024])
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Readout Layer
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(1024, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits



"""
 3x3 conv => 3x3 conv => 2x2 pooling =>
 3x3 conv => 3x3 conv => 2x2 pooling =>
 3x3 conv => FC => FC => FC

 https://arxiv.org/pdf/1409.1556.pdf
"""
def VGG(x):
    mu = 0
    sigma = 0.1  # 0.01

    # 3x3 conv
#     print('----- Stage # 1 -----')
#     print('{} => {}'.format('INPUT', x))
    conv1_W = weight_variable((3, 3, 1, 64), mean=mu, stddev=sigma)
    conv1_b = bias_variable([64])
    conv1 = conv2d(x, conv1_W) + conv1_b
    conv1 = tf.nn.relu(conv1)                       # Input = 32x32x1. Output = 30x30x64.
#     print('{} => {}'.format('RELU 1', conv1))

#     print('----- Stage # 2 -----')
    conv2_W = weight_variable((3, 3, 64, 64), mean=mu, stddev=sigma)
    conv2_b = bias_variable([64])
    conv2 = conv2d(conv1, conv2_W) + conv2_b       # Input = 30x30x64. Output = 28x28x64.
    conv2 = tf.nn.relu(conv2)
#     print('{} => {}'.format('RELU 2', conv2))

    conv2 = max_pool_2x2(conv2)                    # Input = 28x28x64. Output = 14*14*64.
#     print('{} => {}'.format('MAX POOL 2', conv2))  # =============== 1 ====================

#     print('----- Stage # 3 -----')
    conv3_W = weight_variable((3, 3, 64, 128), mean=mu, stddev=sigma)
    conv3_b = bias_variable([128])
    conv3 = conv2d(conv2, conv3_W) + conv3_b       # Input = 14x14x64. Output = 12x12x128.
    conv3 = tf.nn.relu(conv3)
#     print('{} => {}'.format('RELU 3', conv3))

#     print('----- Stage # 4 -----')
    conv4_W = weight_variable((3, 3, 128, 128), mean=mu, stddev=sigma)
    conv4_b = bias_variable([128])
    conv4 = conv2d(conv3, conv4_W) + conv4_b       # Input = 12x12x128. Output = 10x10x128.
    conv4 = tf.nn.relu(conv4)
#     print('{} => {}'.format('RELU 4', conv4))

    conv4 = max_pool_2x2(conv4)                    # Input = 10x10x128. Output = 5*5*128.
#     print('{} => {}'.format('MAX POOL 4', conv4))  # ================= 2 ==================

#     print('----- Stage # 5 -----')
    conv5_W = weight_variable((3, 3, 128, 256), mean=mu, stddev=sigma)
    conv5_b = bias_variable([256])
    conv5 = conv2d(conv4, conv5_W) + conv5_b       # Input = 5*5*128. Output = 3*3*256.
    conv5 = tf.nn.relu(conv5)
#     print('{} => {}'.format('RELU 5', conv5))

#     print('----- Stage # 6 -----')
    conv6_W = weight_variable((3, 3, 256, 256), mean=mu, stddev=sigma)
    conv6_b = bias_variable([256])
    conv6 = conv2d(conv4, conv5_W) + conv5_b       # Input = 3*3*256. Output = 1*1*256.
    conv6 = tf.nn.relu(conv5)
#     print('{} => {}'.format('RELU 6', conv6))

    conv6 = max_pool_2x2(conv6)
#     print('{} => {}'.format('MAX POOL 6', conv6))  # ================= 3 ==================

#     print('----- Stage # 7 -----')
    conv7_W = weight_variable((3, 3, 256, 512), mean=mu, stddev=sigma)
    conv7_b = bias_variable([512])
    conv7 = conv2d(conv6, conv7_W) + conv7_b       # Input = 5*5*128. Output = 3*3*256.
    conv7 = tf.nn.relu(conv7)
#     print('{} => {}'.format('RELU 7', conv7))

#     print('----- Stage # 8 -----')
    conv8_W = weight_variable((3, 3, 512, 512), mean=mu, stddev=sigma)
    conv8_b = bias_variable([512])
    conv8 = conv2d(conv7, conv8_W) + conv8_b       # Input = 3*3*256. Output = 1*1*256.
    conv8 = tf.nn.relu(conv8)
#     print('{} => {}'.format('RELU 8', conv8))

    conv8 = max_pool_2x2(conv8)
#     print('{} => {}'.format('MAX POOL 8', conv8))  # ================== 4 =================

#     print('----- Stage # 9 -----')
#     conv9_W = weight_variable((3, 3, 512, 512), mean=mu, stddev=sigma)
#     conv9_b = bias_variable([512])
#     conv9 = conv2d(conv8, conv9_W) + conv9_b
#     conv9 = tf.nn.relu(conv9)
#     print('{} => {}'.format('RELU 9', conv9))

#     print('----- Stage # 10 -----')
#     conv10_W = weight_variable((3, 3, 512, 512), mean=mu, stddev=sigma)
#     conv10_b = bias_variable([512])
#     conv10 = conv2d(conv9, conv10_W) + conv10_b
#     conv10 = tf.nn.relu(conv10)
#     print('{} => {}'.format('RELU 10', conv10))

#     conv10 = max_pool_2x2(conv10)
#     print('{} => {}'.format('MAX POOL 10', conv10))  # ================== 5 =================

    # FC
#     print('----- FC -----')
    fc0   = flatten(conv8)    # Input = 32*32*512. Output = 32*32*512 => 524,288
#     print('{} => {}'.format('Flatten', fc0))
    fc1_W = weight_variable((2*2*512, 1024),  mean=mu, stddev=sigma)
    fc1_b = bias_variable([1024])
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
#     print('{} => {}'.format('FC 1', fc1))
    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)
#     print('{} => {}'.format('Dropout', fc1))

    # FC 2
    fc2_W = weight_variable((1024, 1024),  mean=mu, stddev=sigma)
    fc2_b = bias_variable([1024])
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
#     print('{} => {}'.format('FC 2', fc2))
    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Readout Layer
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(1024, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits


"""
 3x3 conv => 3x3 conv => 2x2 pooling =>
 3x3 conv => 3x3 conv => 2x2 pooling =>
 3x3 conv => FC => FC => FC

 https://arxiv.org/pdf/1409.1556.pdf
"""
def VGG(x):
    mu = 0
    sigma = 0.1  # 0.01

    # 3x3 conv
#     print('----- Stage # 1 -----')
#     print('{} => {}'.format('INPUT', x))
    conv1_W = weight_variable((3, 3, 1, 64), mean=mu, stddev=sigma)
    conv1_b = bias_variable([64])
    conv1 = conv2d(x, conv1_W) + conv1_b
    conv1 = tf.nn.relu(conv1)                       # Input = 32x32x1. Output = 30x30x64.
#     print('{} => {}'.format('RELU 1', conv1))

#     print('----- Stage # 2 -----')
    conv2_W = weight_variable((3, 3, 64, 64), mean=mu, stddev=sigma)
    conv2_b = bias_variable([64])
    conv2 = conv2d(conv1, conv2_W) + conv2_b       # Input = 30x30x64. Output = 28x28x64.
    conv2 = tf.nn.relu(conv2)
#     print('{} => {}'.format('RELU 2', conv2))

    conv2 = max_pool_2x2(conv2)                    # Input = 28x28x64. Output = 14*14*64.
#     print('{} => {}'.format('MAX POOL 2', conv2))  # =============== 1 ====================

#     print('----- Stage # 3 -----')
    conv3_W = weight_variable((3, 3, 64, 128), mean=mu, stddev=sigma)
    conv3_b = bias_variable([128])
    conv3 = conv2d(conv2, conv3_W) + conv3_b       # Input = 14x14x64. Output = 12x12x128.
    conv3 = tf.nn.relu(conv3)
#     print('{} => {}'.format('RELU 3', conv3))

#     print('----- Stage # 4 -----')
    conv4_W = weight_variable((3, 3, 128, 128), mean=mu, stddev=sigma)
    conv4_b = bias_variable([128])
    conv4 = conv2d(conv3, conv4_W) + conv4_b       # Input = 12x12x128. Output = 10x10x128.
    conv4 = tf.nn.relu(conv4)
#     print('{} => {}'.format('RELU 4', conv4))

    conv4 = max_pool_2x2(conv4)                    # Input = 10x10x128. Output = 5*5*128.
#     print('{} => {}'.format('MAX POOL 4', conv4))  # ================= 2 ==================

#     print('----- Stage # 5 -----')
    conv5_W = weight_variable((3, 3, 128, 256), mean=mu, stddev=sigma)
    conv5_b = bias_variable([256])
    conv5 = conv2d(conv4, conv5_W) + conv5_b       # Input = 5*5*128. Output = 3*3*256.
    conv5 = tf.nn.relu(conv5)
#     print('{} => {}'.format('RELU 5', conv5))

#     print('----- Stage # 6 -----')
    conv6_W = weight_variable((3, 3, 256, 256), mean=mu, stddev=sigma)
    conv6_b = bias_variable([256])
    conv6 = conv2d(conv4, conv5_W) + conv5_b       # Input = 3*3*256. Output = 1*1*256.
    conv6 = tf.nn.relu(conv5)
#     print('{} => {}'.format('RELU 6', conv6))

    conv6 = max_pool_2x2(conv6)
#     print('{} => {}'.format('MAX POOL 6', conv6))  # ================= 3 ==================

#     print('----- Stage # 7 -----')
    conv7_W = weight_variable((3, 3, 256, 512), mean=mu, stddev=sigma)
    conv7_b = bias_variable([512])
    conv7 = conv2d(conv6, conv7_W) + conv7_b       # Input = 5*5*128. Output = 3*3*256.
    conv7 = tf.nn.relu(conv7)
#     print('{} => {}'.format('RELU 7', conv7))

#     print('----- Stage # 8 -----')
    conv8_W = weight_variable((3, 3, 512, 512), mean=mu, stddev=sigma)
    conv8_b = bias_variable([512])
    conv8 = conv2d(conv7, conv8_W) + conv8_b       # Input = 3*3*256. Output = 1*1*256.
    conv8 = tf.nn.relu(conv8)
#     print('{} => {}'.format('RELU 8', conv8))

    conv8 = max_pool_2x2(conv8)
    # FC
#     print('----- FC -----')
    fc0   = flatten(conv8)    # Input = 32*32*512. Output = 32*32*512 => 524,288
#     print('{} => {}'.format('Flatten', fc0))
    fc1_W = weight_variable((2*2*512, 1024),  mean=mu, stddev=sigma)
    fc1_b = bias_variable([1024])
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
#     print('{} => {}'.format('FC 1', fc1))
    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)
#     print('{} => {}'.format('Dropout', fc1))

    # FC 2
    fc2_W = weight_variable((1024, 1024),  mean=mu, stddev=sigma)
    fc2_b = bias_variable([1024])
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
#     print('{} => {}'.format('FC 2', fc2))
    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Readout Layer
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(1024, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits


# 95 %
def LeNet5(x):
    mu = 0
    sigma = 0.1  # 0.01

    # 1st stage
    conv1_W = weight_variable((3, 3, 1, 32), mean=mu, stddev=sigma)
    conv1_b = bias_variable([32])
    conv1 = conv2d(x, conv1_W) + conv1_b
    conv1 = tf.nn.relu(conv1)

    conv12_W = weight_variable((3, 3, 32, 32), mean=mu, stddev=sigma)
    conv12_b = bias_variable([32])
    conv12 = conv2d(conv1, conv12_W) + conv12_b
    conv12 = tf.nn.relu(conv12)

    conv12 = max_pool_2x2(conv12)

    # 2nd stage:
    conv2_W = weight_variable((3, 3, 32, 64), mean=mu, stddev=sigma)
    conv2_b = bias_variable([64])
    conv2 = conv2d(conv12, conv2_W) + conv2_b
    conv2 = tf.nn.relu(conv2)

#     conv21_W = weight_variable((3, 3, 64, 64), mean=mu, stddev=sigma)
#     conv21_b = bias_variable([64])
#     conv21 = conv2d(conv2, conv21_W) + conv21_b
#     conv21 = tf.nn.relu(conv21)
    conv2 = max_pool_2x2(conv2)

    # Fully connected
    fc0   = flatten(conv2)
    fc1_W = weight_variable((2304, 1024),  mean=mu, stddev=sigma)
    fc1_b = bias_variable([1024])
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # FC 2
    fc2_W = weight_variable((1024, 1024),  mean=mu, stddev=sigma)
    fc2_b = bias_variable([1024])
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Readout Layer
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(1024, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits


# 95.4
def LeNet5(x):
    mu = 0
    sigma = 0.1  # 0.01

    # 1st stage
    conv1_W = weight_variable((3, 3, 1, 32), mean=mu, stddev=sigma)
    conv1_b = bias_variable([32])
    conv1 = conv2d(x, conv1_W) + conv1_b
    conv1 = tf.nn.relu(conv1)

    conv12_W = weight_variable((3, 3, 32, 32), mean=mu, stddev=sigma)
    conv12_b = bias_variable([32])
    conv12 = conv2d(conv1, conv12_W) + conv12_b
    conv12 = tf.nn.relu(conv12)

    conv12 = max_pool_2x2(conv12)

    # 2nd stage:
    conv2_W = weight_variable((3, 3, 32, 64), mean=mu, stddev=sigma)
    conv2_b = bias_variable([64])
    conv2 = conv2d(conv12, conv2_W) + conv2_b
    conv2 = tf.nn.relu(conv2)

    conv21_W = weight_variable((3, 3, 64, 64), mean=mu, stddev=sigma)
    conv21_b = bias_variable([64])
    conv21 = conv2d(conv2, conv21_W) + conv21_b
    conv21 = tf.nn.relu(conv21)
#     conv21 = max_pool_2x2(conv21)

    conv22_W = weight_variable((1, 1, 64, 64), mean=mu, stddev=sigma)
    conv22_b = bias_variable([64])
    conv22 = conv2d(conv21, conv22_W) + conv22_b
    conv22 = tf.nn.relu(conv22)
    conv22 = max_pool_2x2(conv22)

    # 3rd stage
#     conv3_W = weight_variable((3, 3, 32, 128), mean=mu, stddev=sigma)
#     conv3_b = bias_variable([64])
#     conv3 = conv2d(conv22, conv3_W) + conv3_b
#     conv3 = tf.nn.relu(conv2)

#     conv31_W = weight_variable((3, 3, 64, 128), mean=mu, stddev=sigma)
#     conv31_b = bias_variable([64])
#     conv31 = conv2d(conv3, conv31_W) + conv21_b
#     conv31 = tf.nn.relu(conv31)

#     conv32_W = weight_variable((1, 1, 64, 128), mean=mu, stddev=sigma)
#     conv32_b = bias_variable([64])
#     conv32 = conv2d(conv31, conv32_W) + conv22_b
#     conv32 = tf.nn.relu(conv32)
#     conv32 = max_pool_2x2(conv32)

#     print(conv32)

    # Fully connected
    fc0   = flatten(conv22)
    fc1_W = weight_variable((1600, 1024),  mean=mu, stddev=sigma)
    fc1_b = bias_variable([1024])
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # FC 2
    fc2_W = weight_variable((1024, 1024),  mean=mu, stddev=sigma)
    fc2_b = bias_variable([1024])
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Readout Layer
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(1024, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits


# http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
# 95.4  <====> 95.8 -- No data generation
def LeNet5(x):
    mu = 0
    sigma = 0.1  # 0.01

    # 1st stage
    conv1_W = weight_variable((3, 3, 1, 32), mean=mu, stddev=sigma)
    conv1_b = bias_variable([32])
    conv1 = conv2d(x, conv1_W) + conv1_b
    conv1 = tf.nn.relu(conv1)

    conv12_W = weight_variable((3, 3, 32, 32), mean=mu, stddev=sigma)
    conv12_b = bias_variable([32])
    conv12 = conv2d(conv1, conv12_W) + conv12_b
    conv12 = tf.nn.relu(conv12)

    conv12 = max_pool_2x2(conv12)

    # 2nd stage:
    conv2_W = weight_variable((3, 3, 32, 64), mean=mu, stddev=sigma)
    conv2_b = bias_variable([64])
    conv2 = conv2d(conv12, conv2_W) + conv2_b
    conv2 = tf.nn.relu(conv2)

    conv21_W = weight_variable((3, 3, 64, 64), mean=mu, stddev=sigma)
    conv21_b = bias_variable([64])
    conv21 = conv2d(conv2, conv21_W) + conv21_b
    conv21 = tf.nn.relu(conv21)

    conv22_W = weight_variable((1, 1, 64, 64), mean=mu, stddev=sigma)
    conv22_b = bias_variable([64])
    conv22 = conv2d(conv21, conv22_W) + conv22_b
    conv22 = tf.nn.relu(conv22)
    conv22 = max_pool_2x2(conv22)

    # 3rd stage
    conv3_W = weight_variable((3, 3, 64, 64), mean=mu, stddev=sigma)
    conv3_b = bias_variable([64])
    conv3 = conv2d(conv22, conv3_W) + conv3_b
    conv3 = tf.nn.relu(conv3)

    # Fully connected
    fc0   = flatten(conv3)
    fc1_W = weight_variable((3*3*64, 512),  mean=mu, stddev=sigma)
    fc1_b = bias_variable([512])
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # FC 2
    fc2_W = weight_variable((512, 512),  mean=mu, stddev=sigma)
    fc2_b = bias_variable([512])
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Readout Layer
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(512, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits


def LeNet5(x):
    mu = 0
    sigma = 0.1  # 0.01

    # 1st stage
    conv1_W = weight_variable((3, 3, 1, 32), mean=mu, stddev=sigma)
    conv1_b = bias_variable([32])
    conv1 = conv2d(x, conv1_W) + conv1_b
    conv1 = tf.nn.relu(conv1)

    conv12_W = weight_variable((3, 3, 32, 32), mean=mu, stddev=sigma)
    conv12_b = bias_variable([32])
    conv12 = conv2d(conv1, conv12_W) + conv12_b
    conv12 = tf.nn.relu(conv12)

    conv12 = max_pool_2x2(conv12)

    # 2nd stage:
    conv2_W = weight_variable((3, 3, 32, 64), mean=mu, stddev=sigma)
    conv2_b = bias_variable([64])
    conv2 = conv2d(conv12, conv2_W) + conv2_b
    conv2 = tf.nn.relu(conv2)

    conv21_W = weight_variable((3, 3, 64, 64), mean=mu, stddev=sigma)
    conv21_b = bias_variable([64])
    conv21 = conv2d(conv2, conv21_W) + conv21_b
    conv21 = tf.nn.relu(conv21)

    conv22_W = weight_variable((1, 1, 64, 64), mean=mu, stddev=sigma)
    conv22_b = bias_variable([64])
    conv22 = conv2d(conv21, conv22_W) + conv22_b
    conv22 = tf.nn.relu(conv22)
    conv22 = max_pool_2x2(conv22)

    # 3rd stage
    conv3_W = weight_variable((3, 3, 64, 64), mean=mu, stddev=sigma)
    conv3_b = bias_variable([64])
    conv3 = conv2d(conv22, conv3_W) + conv3_b
    conv3 = tf.nn.relu(conv3)

    # Fully connected
    fc0   = flatten(conv3)
    fc1_W = weight_variable((3*3*64, 512),  mean=mu, stddev=sigma)
    fc1_b = bias_variable([512])
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # FC 2
    fc2_W = weight_variable((512, 256),  mean=mu, stddev=sigma)
    fc2_b = bias_variable([256])
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Readout Layer
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(256, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits

# 96.3
def LeNet5(x):
    mu = 0
    sigma = 0.1  # 0.01

    # 1st stage
    conv1_W = weight_variable((3, 3, 1, 64), mean=mu, stddev=sigma)
    conv1_b = bias_variable([64])
    conv1 = conv2d(x, conv1_W) + conv1_b
    conv1 = tf.nn.relu(conv1)

    conv12_W = weight_variable((3, 3, 64, 64), mean=mu, stddev=sigma)
    conv12_b = bias_variable([64])
    conv12 = conv2d(conv1, conv12_W) + conv12_b
    conv12 = tf.nn.relu(conv12)

    conv12 = max_pool_2x2(conv12)

    # 2nd stage:
    conv2_W = weight_variable((3, 3, 64, 128), mean=mu, stddev=sigma)
    conv2_b = bias_variable([128])
    conv2 = conv2d(conv12, conv2_W) + conv2_b
    conv2 = tf.nn.relu(conv2)

    conv21_W = weight_variable((3, 3, 128, 128), mean=mu, stddev=sigma)
    conv21_b = bias_variable([128])
    conv21 = conv2d(conv2, conv21_W) + conv21_b
    conv21 = tf.nn.relu(conv21)

    conv22_W = weight_variable((1, 1, 128, 128), mean=mu, stddev=sigma)
    conv22_b = bias_variable([128])
    conv22 = conv2d(conv21, conv22_W) + conv22_b
    conv22 = tf.nn.relu(conv22)
    conv22 = max_pool_2x2(conv22)

    # 3rd stage
    conv3_W = weight_variable((3, 3, 128, 128), mean=mu, stddev=sigma)
    conv3_b = bias_variable([128])
    conv3 = conv2d(conv22, conv3_W) + conv3_b
    conv3 = tf.nn.relu(conv3)

    # Fully connected
    fc0   = flatten(conv3)
    fc1_W = weight_variable((3*3*128, 512),  mean=mu, stddev=sigma)
    fc1_b = bias_variable([512])
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # FC 2
    fc2_W = weight_variable((512, 256),  mean=mu, stddev=sigma)
    fc2_b = bias_variable([256])
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Readout Layer
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(256, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits
