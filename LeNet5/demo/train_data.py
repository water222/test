import tensorflow as tf
import numpy as np
from demo.Data_handle import dataprovider
from demo.LetNet5 import LeNet5


class Trainner():
    '''
    batch_size:需要与LetNet5.py中的batch_size保持一致
    self.xs:需要与LetNet5.py中的self.xs保持一致
    self.ys：需要与LetNet5.py中的self.ys保持一致
    return:训练好参数的模型，以便测试可以直接用上
    '''
    def __init__(self,xs_p,ys_p,images,labels,predict,loss,optimizer,epoches,batch_size=64,mapsize=32,channels=3):
        self.images = images
        self.labels = labels
        self.predict = predict
        self.loss = loss
        self.optimizer = optimizer
        self.epoches = epoches
        self.batch_size = batch_size

        # self.xs = tf.placeholder(dtype=tf.float32,shape=[None, mapsize, mapsize, channels],name='images')#是不是可以不定xs的尺寸？
        # self.ys = tf.placeholder(dtype=tf.float32,shape=[None,10],name='labels')
        self.xs = xs_p#这两个赋值至关重要，不然train时找不到已经填好的坑在哪里！此步骤对测试也同样适用
        self.ys = ys_p

        #保存模型
        self.save_path = 'model/default.ckpt'
        # self.saver = None

    def get_trainchunk(self):
        with tf.name_scope('get_trainchunk'):
            '''
           作用相当于迭代器，用于批量提取数据
           batch_size 通常为自己设定
            '''
            if len(self.labels) != len(self.images):
                raise Exception('Length of images and labels is unequal..')
            stepStart = 0
            # i = 0
            while stepStart < len(self.images):
                stepEnd = stepStart + self.batch_size
                if stepEnd < len(self.images):
                    yield self.images[stepStart:stepEnd], self.labels[stepStart:stepEnd]
                    # i += 1
                stepStart = stepEnd

        # 计算训练的准确度

    def Getaccuracy(self, predictions, orlabel):
        with tf.name_scope('accuracy'):
            with tf.Session() as sess:
                correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(orlabel, 1))  # 评估模型的correct_prediction
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))  # 计算精确度
                return sess.run(accuracy)

    # 训练网络
    def train(self):
        with tf.name_scope('train'):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print('start training - - - - - -')
                i = 0
                # all_predictions = []
                while (self.epoches):  # 控制迭代几次
                    self.epoches -= 1
                    for samples, labels in self.get_trainchunk():  # 控制循环的批次该函数用于提取批量数据
                        batch_xs, batch_ys = samples, labels  # 这里是如何进行随机抽取的？
                        # print(batch_ys)
                        predictions,loss_,optimizer_ = sess.run([self.predict, self.loss, self.optimizer],
                                                                feed_dict={self.xs: batch_xs,self.ys: batch_ys})  # 这里真的是调试的血泪史！！一定要知道这里的predictions输出的是个list，且长度是3，它真正的结果是要result[0]
                        if i % 10 == 0:  # 每训练10个batch_size,则输出一次准确率
                            predictions = np.array(predictions)  # run之后得到的predictions是一个列表，必须要将它转为数组才能进行接下来的操作
                            # print(predictions.shape)
                            # print(batch_ys.shape)
                            acuracy = self.Getaccuracy(predictions, batch_ys)
                            print('acuracy', acuracy)  # 返回的acuracy是一个tensor如果不用会话驱动的话
                            i += 1
                        # print(predictions,loss_,optimizer_)
                        break

                # 保存网络模型
                # 检查要存放的路径值否存在。这里假定只有一层路径。
                saver = tf.train.Saver()  # saver 不可以定义在初始化函数中，我也不知道为什么
                import os
                if os.path.isdir(self.save_path.split('/')[0]):
                    save_path = saver.save(sess, self.save_path)
                    print("Model saved in file: %s" % save_path)
                else:
                    os.makedirs(self.save_path.split('/')[0])
                    save_path = saver.save(sess, self.save_path)
                    print("Model saved in file: %s" % save_path)
                return saver
                # self.saver = saver

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

    batch_size = 64
    #导入数据
    trainImgs, trainLabels = Gettraindata()
    # 构建网络的可用性

    net = LeNet5('adam', 0.01)  # 需要输入优化方法和学习率
    y_predict, loss, optimizer,xs_p,ys_p = net.create_LeNet5()  # 返回预测值、loss值

    #训练网络需要输入参数：images, labels, predict, loss, optimizer, epoches
    epoches = 1
    trainer = Trainner(xs_p,ys_p,trainImgs,trainLabels,y_predict,loss,optimizer,epoches)
    saver = trainer.train()