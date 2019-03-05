import tensorflow as tf
import numpy as np
from demo.Data_handle import dataprovider
from demo.LetNet5 import LeNet5
from demo.train_data import Trainner
from demo.train_data import Gettraindata

def Gettestdata():
    # 导入测试数据
    print('the Testdata is loading...')
    path = "../data/test_32x32.mat"
    data_provider = dataprovider(path)
    data_provider.loaddata()
    # print('the distribution of data is showing....')
    # data_provider.distribution()#显示数据的分布图
    test_images, test_labels = data_provider.reformat()
    if test_images is not None:
        print('the Testdata loading is successfully!\n')
    return test_images,test_labels


class Test():
    def __init__(self,testimages,testlabels,yt_predict,yt_loss,yt_optimizer,xt_p,yt_p,saver,batch_size=64):
        self.testimgages = testimages
        self.testlabels = testlabels
        self.batch_size = batch_size
        self.predict = yt_predict
        self.loss = yt_loss
        self.optimizer = yt_optimizer
        self.saver = saver

        self.xs = xt_p
        self.ys = yt_p


    def get_testchunk(self):
        with tf.name_scope('get_testchunk'):
            '''
           作用相当于迭代器，用于批量提取数据
           batch_size 通常为自己设定
            '''
            if len(self.testlabels) != len(self.testimgages):
                raise Exception('Length of images and labels is unequal..')
            stepStart = 0
            # i = 0
            while stepStart < len(self.testimgages):
                stepEnd = stepStart + self.batch_size
                if stepEnd < len(self.testimgages):
                    yield self.testimgages[stepStart:stepEnd], self.testlabels[stepStart:stepEnd]
                    # i += 1
                stepStart = stepEnd

    def test(self):
        with tf.name_scope('train'):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print('\nstart testing - - - - - -')
                # i = 0
                # all_predictions = []

                saver = self.saver
                saver.restore(sess,'model/default.ckpt')


                for samples, labels in self.get_testchunk():  # 控制循环的批次该函数用于提取批量数据
                    batch_xs, batch_ys = samples, labels  # 这里是如何进行随机抽取的？
                    predictions, loss_, optimizer_ = sess.run([self.predict, self.loss, self.optimizer],
                                                              feed_dict={self.xs: batch_xs,
                                                                         self.ys: batch_ys})  # 这里真的是调试的血泪史！！一定要知道这里的predictions输出的是个list，且长度是3，它真正的结果是要result[0]
                    # if i % 10 == 0:  # 每训练10个batch_size,则输出一次准确率
                    #     predictions = np.array(predictions)  # run之后得到的predictions是一个列表，必须要将它转为数组才能进行接下来的操作
                    #     # print(predictions.shape)
                    #     # print(batch_ys.shape)
                    #     acuracy = self.Getaccuracy(predictions, batch_ys)
                    #     print('acuracy', acuracy)  # 返回的acuracy是一个tensor如果不用会话驱动的话
                    #     i += 1
                    # print(self.predict)
                    # print(self.xs)

                    print(predictions,loss_,optimizer_)
                    break


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

    #测试网络
    testImgs, testLabels = Gettestdata()
    # print(testImgs.shape)

    # testimages, testlabels, yt_predict, yt_loss, yt_optimizer,xt_p, yt_p,
    net = LeNet5('adam', 0.01)
    yt_predict, yt_loss, yt_optimizer,xt_p, yt_p = net.create_LeNet5()
    # print(yt_predict)
    test_ = Test(testImgs,testLabels,yt_predict, yt_loss, yt_optimizer,xt_p, yt_p,saver)
    test_.test()