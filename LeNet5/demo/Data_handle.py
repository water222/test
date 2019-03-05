#  搭建卷积神经网络、规范代码、体会with name_scope、Summary TensorBoard的好处
#比较卷积神经网路和U-net的损失函数有何不同。
import tensorflow as tf
import numpy as np
from scipy.io import loadmat as load#python 用于导入.mat文件的函数
import matplotlib.pyplot as plt

class dataprovider():
    def __init__(self,path):
        self.path = path

    def loaddata(self):
        # path = "../data/train_32x32.mat"
        train =load(self.path)#该文件夹中包含了训练数据与其对应的标签
        self.images = train['X']
        self.labels = train['y']
        if self.images is None:
            print('Images has no found..')
        else:
            print('Images has found successfully！')
        if self.labels is None:
            print('labels has no found..')
        else:
            print('labels has found successfully！')
        # print('Train datas shape',self.images.shape)
        # print('Train labels shape',self.labels.shape)
        #数据集中X,y的大小写要根据该数据集的介绍而定
        return self.images,self.labels

    def reformat(self):
        '''
        1.导入的mat文件格式[imagex,imagey,channel,batchsize]
        2.python默认文件格式[batchsize,imagex,imagey,channel]
        3.因此要对.mat文件格式进行处理
        4.将label变成one-hot模式（关于该模式的定义参考标签“机器学习理论解释”）
        5.对数据的处理通常分为将二维的图片reshape为四维（如果直接下载的“知名”数据集，这部分工作可能已经完成了）、
        或者是先探索数据集是否符合该编译器或者编程语言的默认规则。
        '''
        # print('Train images format is changing - - - - - - -')
        self.images = np.transpose(self.images,(3,0,1,2))#改变图片的格式
        # print('R_Train datas shape', self.images.shape)

        #将labels变成one-hot模式
        # print('Train labels format is changing to one-hot - - - - - - -')
        self.labels = np.array([x[0] for x in self.labels])
        #x表示的是一个1*1的list[],而x[0]表示的是一个数，上述操作将73257*1的矩阵变成1*73257的矩阵
        one_hot_labels= []
        for label in self.labels:
            # print(label.shape)
            one_hot = [0.0]*10 #创建一个1*10的矩阵，因为label是10维的
            if label == 10:
                one_hot[0] = 1.0#.mat文件中是1-10，故需要将10变成0
            else:
                one_hot[label] = 1.0
            one_hot_labels.append(one_hot)
        self.labels = np.array(one_hot_labels)
        # print('Train labels has change successfully,the 100th label is :',self.labels[100])
        return self.images,self.labels

    def imageshow(self,count):
        '''显示图片,在显示图片前需要先将图片格式reformat
        images: 要显示的图片集
        labels: 图片所对应的标签，主要用于检测显示的图片是否是正确的
        count：显示图片集中的第几张图片
        '''
        # self.reformat()
        print('the photo will show the number:',self.labels[count])
        plt.imshow(self.images[count])
        print('即将弹窗显示图片--------')
        plt.show()

    def distribution(self):
        '''
        查看一下每个label，即0-9的分布情况，并以条形图的形式统计出来
        需要注意的是显示分布条形图时不能进行reformat处理，与imageshow不同
        '''
        # self.loaddata()
        print('the labels distribution will show --------')
        count={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
        for label in self.labels:
            label[0] = 0 if label[0]== 10 else label[0]
            count[str(label[0])] += 1#注意要将label[0]转为字符串
        print('labels and numbers of labels:',count)#打印标签及其对应个数

        count = count.items()#将字典转换成列表
        print('dict change to list....：',count)
        x = []
        y = []
        for c in count:
            x.append(c[0])
            y.append(c[1])

        pos = np.arange(len(x))
        plt.bar(pos,y, align='center',alpha=0.5)#0.5表示条形的宽度
        plt.title('Label Distribution')
        plt.show()

if __name__ == '__main__':
    path = "../data/train_32x32.mat"
    data = dataprovider(path)
    data.loaddata()#导入数据集
    data.reformat()#处理数据集格式
    data.imageshow(68)#显示第i张图片，显示图片前需要进行reformat处理
    # data.distribution()#显示每个标签的分布条形图，显示分布条形图时不能进行reformat处理
