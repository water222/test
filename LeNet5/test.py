import tensorflow as tf
import numpy as np

# a = tf.placeholder(dtype=tf.float32,shape = (2,1))
# # a = [3,4]
# print(a.shape)
# print(a)
# a = np.array([x for x in a])
# print(a.shape)
# print(a)
# labels=[1,2,2,34,4,3,5,6,78]
# count = {}
# for label in labels:
#     if label in count:
#         count[label] += 1
#     else:
#         count[label] = 1
# print(count)
# for k,v in count.items():
#     print(k,v)
# labels=[[1],[2],[2],[3],[4],[4],[3],[5],[6],[7],[8]]
# count = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
# count['0'] += 1
# print(count)
#
# from demo.Data_handle import dataprovider

# import tensorflow as tf
# from demo.Data_handle import dataprovider
# import matplotlib.pyplot as plt
# import imghdr
#
# with tf.Session() as sess:
#     data = dataprovider('data/train_32x32.mat')
#     data.loaddata()
#     data_images,data_labels = data.reformat()#如果这里不用两个变量接收的话，那么data_images包含的则是图像和标签
#     # img = tf.read_file()
#     # imgType = imghdr.what(data_images[0])查看图片类型，不过在此报错
#     img_datas=[]
#     for i in range(3):
#         # plt.imshow(data_images[i])
#         # plt.show()
#         # img_data = tf.image.decode_png(data_images[i],channels=1)
#         img_data = sess.run(tf.image.rgb_to_grayscale(data_images[i]))
#         plt.imshow('changed',img_data)
#         plt.show()
#         img_datas.append(img_data)

    # print(img_datas.shape)
#
# import tensorflow as tf
# import numpy as np
# c=tf.constant(value=1)
# #print(assert c.graph is tf.get_default_graph())
# print(c.graph)
# # print(tf.get_default_graph())
#
#
# g=tf.Graph()
# print("g:",g)
# with g.as_default():
#     d=tf.constant(value=2)
#     print(d.graph)
#     #print(g)


# import tensorflow as tf
# # import tensorflow.examples.tutorials.mnist.input_data as input_data
# # data_set = input_data.read_sets()
# batch_size = data_set.nex
#


#Summaries的用法
# merged_summary_op = tf.merge_all_summaries()
# summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)
# total_step = 0
# while training:
#   total_step += 1
#   session.run(training_op)
#   if total_step % 100 == 0:
#     summary_str = session.run(merged_summary_op)
#     summary_writer.add_summary(summary_str, total_step)

#
#
# #checkpoint的保存
# import tensorflow as tf
# # Create some variables.
# # v1 = tf.placeholder(dtype= tf.float32,shape=[1,4], name="v1")
# # v2 = tf.placeholder(dtype= tf.float32,shape=[1,4], name="v2")placeholder必须要在run方法中feed值
#
# v1 = tf.Variable([1,4], name="v1")
# v2 = tf.Variable([1,4], name="v2")
#
# # Add an op to initialize the variables.
# init_op = tf.global_variables_initializer()
#
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
# with tf.Session() as sess:
#   sess.run(init_op)
#   # v1=[1,2,3,4]
#   # v2=[1,3,5,7]
#   v3 = tf.add(v1,v2)
#   print('v3 is :',v3)
#   print('sess_v3 is :',sess.run(v3))
#
#   # Save the variables to disk.
#   save_path = saver.save(sess, "/tmp/model.ckpt")
#   print("Model saved in file: ", save_path)
# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#   # Restore variables from disk.
#   # saver.restore(sess, "/tmp/model.ckpt")#恢复模型
#   print("Model restored.")
#   v4 = tf.Variable([1, 4], name="v4")
#   sess.run(tf.global_variables_initializer())
#
#   # v4 = [1, 3, 5, 7]
#   v5 = tf.add(v4,v3)
#   print('sess_v5',sess.run(v5))

import numpy as np
import tensorflow as tf

a = [[0,1,0,0],[1,0,0,0],[0,0,0,1]]
b = [[0,1,0,0],[0,1,0,0],[0,3,0,1]]
a = np.array(a)
b = np.array(b)
# print(a.shape)
a = np.argmax(a,1)
b = np.argmax(b,1)
# print(a.shape)
print(a)
print(b)

c = tf.equal(a,b)
d = tf.cast(c,'float')
e = tf.reduce_mean(d)
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(e))