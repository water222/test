import tensorflow as tf
import numpy as np
from demo.Data_handle import dataprovider

'''
    网络需要输入的是：学习方法、学习速率
    
    网络定义的是：x.shape=[64,32,32,3]、y.shape=[64,10]
    还有本网络的卷积核大小为5x5，步长1，池化2x2,步长为2
    上述定义有任一改变了需要针对修改
    
    return:预测值y_predict、loss、optimizer
'''
class LeNet5():
    def __init__(self, optimizeMethod, learning_rate, kernel_size=5, outputs=32, keep_prob=0.75):
        #网络需要的外界输入
        self.kernel_size = kernel_size
        self.outputs = outputs
        self.keep_prob = keep_prob
        self.optimizeMethod = optimizeMethod
        self.learning_rate = learning_rate

        # self.xs = tf.placeholder(dtype=tf.float32, shape=[64, 32, 32, 3])
        # self.ys = tf.placeholder(dtype=tf.float32, shape=[64, 10])
        #网络的输出
        self.y_predict = None
        self.loss = None
        self.optimizer = None

    def weight_variable(self, shape, stddev=0.1, name='weight'):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, stddev=0.1, name='bias'):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self,x,w,b):
        with tf.name_scope('conv2d'):
            conv_2d = tf.nn.conv2d(x,w,strides = [1,1,1,1],padding='VALID')
            conv_2d = tf.nn.bias_add(conv_2d,b)
            return tf.nn.relu(conv_2d)

    def max_pool2x2(self,x):
        with tf.name_scope('maxpool2x2'):
            return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding = 'VALID')

    def create_LeNet5(self):
        '''
        LeNet5卷积神经网络中，使用的卷积核大小为5*5，步长为strides=[1,1,1,1]
        pooling_size=2*2,pooling_strides=[1,2,2,1]
        :return:y_predict
        '''
        with tf.name_scope('creat_LeNet5'):

            layers = 2
            x = tf.placeholder(dtype = tf.float32, shape = [64, 32, 32, 3])
            self.x = x
            self.y = tf.placeholder(dtype= tf.float32, shape = [64, 10])

            # 卷积层和池化层
            for layer in range(layers):
                if layer == 0:
                    w = self.weight_variable(shape=[self.kernel_size, self.kernel_size,
                                                    3, self.outputs], name='w')
                    b = self.bias_variable(shape=[self.outputs], name='w')
                else:
                    w = self.weight_variable(shape=[self.kernel_size, self.kernel_size,
                                                    self.outputs, self.outputs * 2], name='w')  # 卷积操作，卷积之后是relu
                    b = self.bias_variable(shape=[self.outputs * 2], name='b')
                    self.outputs = self.outputs * 2

                # print(x.shape,w.shape,b.shape)

                conv = self.conv2d(x, w, b)  # 返回的是已经卷积和relu后的图像
                print('conv_layer %s has compelete,image shape has change to:' % layer, conv.shape)
                poolayer = self.max_pool2x2(conv)  # 池化操作
                print('pool_layer %s has compelete,image shape has change to:' % layer, poolayer.shape)
                x = poolayer
            print('we are going to fully connection layer - - - - - - -')

            # 全连接层
            a = int((x.shape[1])) ** 2 * int(x.shape[3])
            w_fc1 = self.weight_variable([a, 533])  # [5*5*64,533]
            b_fc1 = self.bias_variable([533])  # [533]
            # print('before reshape,the x shape is:', x.shape)
            reshape_x = tf.reshape(x, [-1, a])  # [28,25*25*64]
            # print('after reshape, the shape is:', reshape_x.shape)
            # print('the shape of w_fc1 is:', w_fc1.shape)
            h_fc1 = tf.nn.relu(tf.matmul(reshape_x, w_fc1) + b_fc1)

            # dropout为了减少过拟合，我们在输出层之前加入dropout。
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
            # print('h_fc1_drop is :', h_fc1_drop)

            # outputmap
            w_fc2 = self.weight_variable([int(h_fc1_drop.shape[1]), 10])
            b_fc2 = self.bias_variable([10])
            y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
            self.y_predict = y_predict
            self.loss = self.getCost()
            self.train_step = self.Getoptimizer()
            print('y_predict is :', y_predict)
            print('loss is :', self.loss)
            print('train_step is :\n', self.train_step)

            return self.y_predict,self.loss,self.train_step,self.x,self.y

    def getCost(self):
        '''
        计算损失函数，需要输入网络的predict值
        将会返回损失函数:交叉熵
        ps:交叉熵是一个‘完美’的loss函数
        而本例中的self.optimizer是系统自动地使用‘你’选择的学习算法来最小化成本
        :return:loss
        '''
        with tf.name_scope('getCost'):
            cross_entropy = -tf.reduce_sum(self.y * tf.log(self.y_predict))
            return cross_entropy

    def Getoptimizer(self):
        '''
        计算optimizer？作用是什么
        而本例中的self.optimizer是系统自动地使用‘你’选择的学习算法来最小化成本
        :return:train_step
        '''
        if (self.optimizeMethod == 'gradient'):
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        elif (self.optimizeMethod == 'momentum'):
            train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.5).minimize(self.loss)
        elif (self.optimizeMethod == 'adam'):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        return train_step

def Gettraindata():
    # 导入训练数据
    print('the data is loading...')
    path = "../data/train_32x32.mat"
    data_provider = dataprovider(path)
    data_provider.loaddata()
    # print('the distribution of data is showing....')
    # data_provider.distribution()#显示数据的分布图
    train_images, train_labels = data_provider.reformat()
    if train_images is not None:
        print('the data loading is successfully!\n')
    return train_images,train_labels

if __name__ == '__main__':
    # 测试网络的可用性
    net = LeNet5('adam',0.01)#需要输入优化方法和学习率
    y_predict,loss,optimizer,xs,y = net.create_LeNet5()#返回预测值、loss值、optimizer、x、y占位符所在的位置？
    # print(y_predict,loss,optimizer,xs,y)

