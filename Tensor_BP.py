'''
Created on 2020年9月9日

@author: xiaoyw
'''
import numpy as np
import tensorflow as tf
import pandas as pd
#数据源：以专家投票频次与评分标准向量积算出指标评分，与PCE保持一致。
def get_DataFromExcel():
    df = pd.read_excel('BP.xlsx') 

    return df
# 生成样例数据集
def generate_data():
    data = get_DataFromExcel()
    del_name = ['总评价']
    #拆分特征与标签
    labels = data['总评价'].values/100
    
    labels = labels.reshape(labels.shape[0],-1)
    inputs = data.drop(del_name,axis = 1).values
       
    return inputs, labels

#定义神经元
def NN(h_in,h_out,layer='1'):
    w = tf.Variable(tf.truncated_normal([h_in,h_out],stddev=0.1),name='weights' +layer )
    b = tf.Variable(tf.zeros([h_out],dtype=tf.float32),name='biases' + layer)
    
    return w,b

#定义BP神经网络
def BP_NN(in_units,layers=[10,5,1],dropout=True):
    #定义输入变量
    x = tf.placeholder(dtype=tf.float32,shape=[None,in_units],name='x')
    num = len(layers)   # 网络层数
    #定义网络参数
    w1,b1 = NN(in_units,layers[0],'1')   #定义第一层参数
    w2,b2 = NN(layers[0],layers[1],'2')   #定义第二层参数
    #定义网络隐藏层
    #定义前向传播过程
    h1 = tf.nn.tanh(tf.add(tf.matmul(x,w1),b1))
    #定义dropout保留的节点数量
    keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
    if dropout:
            #使用dropout
            h1_drop = tf.nn.dropout(h1,rate = 1 - keep_prob)        #Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    else:
        h1_drop = h1
    # 针对三层网络
    if num > 2:
        w3,b3 = NN(layers[1],layers[2],'3')   #定义第三一层参数
        # 定义第二层隐藏层
        h2 = tf.nn.tanh(tf.add(tf.matmul(h1_drop,w2),b2))
        #h2 = tf.add(tf.matmul(h1_drop,w2),b2)
        # 定义输出层
        y_conv = tf.nn.sigmoid(tf.add(tf.matmul(h2,w3),b3),name='y_conv') 
    else:
        y_conv = tf.nn.sigmoid(tf.add(tf.matmul(h1_drop,w2),b2),name='y_conv') 
        #y_conv = tf.add(tf.matmul(h1_drop,w2),b2) 

    #定义输出变量    
    y_ = tf.placeholder(dtype=tf.float32,shape=[None,layers[num - 1]],name='y_')

    #定义损失函数及反向传播方法。
    loss_mse = tf.reduce_mean(tf.square(y_conv-y_)) 
    #均方误差MSE损失函数
    train_step = tf.train.GradientDescentOptimizer(0.002).minimize(loss_mse)    
    correct_pred = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    
    return x,y_,loss_mse,train_step,correct_pred,keep_prob

def BPNN_train():
    inputs, labels = generate_data()
    #print(inputs)
    #print(labels)
    # 定义周期、批次、数据总数、遍历一次所有数据需的迭代次数
    n_epochs = 3
    batch_size = 6
    
    # 使用from_tensor_slices将数据放入队列，使用batch和repeat划分数据批次，且让数据序列无限延续
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.batch(batch_size).repeat()
    
    # 使用生成器make_one_shot_iterator和get_next取数据
    iterator = dataset.make_one_shot_iterator()
    next_iterator = iterator.get_next()
    
    #定义神经网络的参数
    in_units = 16  #输入16个指标，返回一个评分
        
    # 定义三层BP神经网络，层数及神经元个数通过layers参数确定，两层[5,3]，只支持2或3层，其他无意义
    x,y_,loss_mse,train_step,correct_pred,keep_prob = BP_NN(in_units,layers=[8,8,1],dropout=False)
    
    saver = tf.train.Saver()  #定义saver
    #随机梯度下降算法训练参数
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for i in range(10000):
            batch_x,batch_y = sess.run(next_iterator)
    
            _,total_loss = sess.run([train_step,loss_mse], feed_dict={x:batch_x,y_:batch_y,keep_prob:0.8})
            
            if i%100 == 0:
                #train_accuracy = accuracy.eval(session = sess,feed_dict={x:batch_x, y_: batch_y, keep_prob: 1.0}) # 用于分类识别，判断准确率
                #print ("step {}, training accuracy {}".format(i, train_accuracy))
                print ("step {}, total_loss {}".format(i, total_loss))           # 用于趋势回归，预测值
            
        saver.save(sess, 'save/BP_model.ckpt') #模型储存位置

# 根据输入预测结果
def BPNN_Pprediction(input_x):    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('save/BP_model.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('save/'))
      
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")
        # 输出预测结果
        y_conv = graph.get_tensor_by_name('y_conv:0')
        keep_prob = graph.get_tensor_by_name("keep_prob:0")

        ret = sess.run(y_conv, feed_dict={x:input_x,keep_prob:1.0})
        #y = sess.run(tf.argmax(ret,1))  # 用于分类问题，取最大概率
        print("预测结果：{}".format(ret[0]))

def main():
    train = True
    #train = False
    
    if train:
        BPNN_train()
    
    input_x = [[0.54,0.56,0.78,0.44,0.48,0.72,0.72,0.54,0.44,0.68,0.68,0.8,0.84,0.84,0.74,0.8]]
    BPNN_Pprediction(input_x)
    
    return
        

if __name__ == '__main__':
    main()
    
