from cv2 import cv2
import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout
import matplotlib.pyplot as plt  # 绘图
from collections import Counter  # 统计频数并且返回字典
import torch
from torch import nn
from torch.nn import ReLU, Sequential, Dropout


class prenn(nn.Module):
    def __init__(self):
        super(prenn, self).__init__()

        # def __init__(self):
        #     super(pre_nn, self).__init__()
        self.model1 = Sequential(
            ReLU(),
            Dropout(p=0.01),
            ReLU(),
            Dropout(p=0.01),
            ReLU(),
            Dropout(p=0.01),
            ReLU(),
            Dropout(p=0.01),
            ReLU(),
            Dropout(p=0.01),
            ReLU(),
            Dropout(p=0.01),
        )

    def forward(self, input):
        return self.model1(input)


def predivt_nn(data, label):
    data = torch.from_numpy(data)
    print('data', data)
    pnn = prenn()
    pnn.train()
    output = pnn(data)
    loss_fn = nn.MSELoss()
    print('output', output)
    optimizer = torch.optim.Adam(pnn.parameters(), lr=0.01)
    loss = loss_fn(output, label)
    loss.requires_grad_(True)
    # 优化器优化模型
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return pnn


# ---------------------------------here any img input would transform to data with satisfactory form---------------
def img_division(img, flag):
    # print(img)
    row, col = img.shape;
    lc = np.zeros((row, col));
    data = [];
    label = [];
    lc = []
    # row,col获取图像的行和列；lc先定义一个和图像尺寸相同的0矩阵；data ，label 为空列表
    for r in range(1, row - 2):
        for c in range(1, col - 2):
            v1 = int(img[r - 1][c]);
            v2 = int(img[r][c - 1]);
            v3 = int(img[r + 1][c]);
            v4 = int(img[r][c + 1])  # 像素上下左右四个像素
            x = int(img[r][c]);
            w1 = int(img[r - 1][c + 2]);
            w2 = int(img[r][c + 2]);
            w3 = int(img[r + 1][c - 1])
            w4 = int(img[r + 1][c + 1]);
            w5 = int(img[r + 1][c + 2]);
            w6 = int(img[r + 2][c - 1]);
            w7 = int(img[r + 2][c]);
            w8 = int(img[r + 2][c + 1]);
            wL = int(img[r + 2][c + 2])
            # w1到wL为像素周围的九个圈的像素
            f0 = abs(v2 - w3);
            f1 = abs(w3 - w6);
            f2 = abs(v3 - w7);
            f3 = abs(v4 - w4);
            f4 = abs(w4 - w8);
            f5 = abs(w1 - w2);
            f6 = abs(w2 - w5);
            f7 = abs(w5 - wL)
            f8 = abs(v4 - w2);
            fL = abs(w3 - v3);
            f_0 = abs(v3 - w4);
            f_1 = abs(w4 - w5);
            f_2 = abs(w6 - w7);
            f_3 = abs(w7 - w8);
            f_4 = abs(w8 - wL)
            D = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + fL + f_0 + f_1 + f_2 + f_3 + f_4  # D为论文中像素复杂度c
            # D1=f0+fL+f3+f_0
            '''s = np.mean([f0,f1,f2,f3,f4,f5,f6,f7,f8,fL,f_0,f_1,f_2,f_3,f_4])
            s1 = math.pow((f0-s),2);s2 = math.pow((f1-s),2);s3 = math.pow((f2-s),2);s4 = math.pow((f3-s),2);s5 = math.pow((f4-s),2)
            s6 = math.pow((f5 - s), 2);s7 = math.pow((f6-s),2);s8 = math.pow((f7-s),2);s9 = math.pow((f8-s),2);s10 = math.pow((fL-s),2)
            s11 = math.pow((f_0 - s), 2);s12 = math.pow((f_1-s),2);s13 = math.pow((f_2-s),2);s14 = math.pow((f_3-s),2);s15 = math.pow((f_4-s),2)
            ave_s = np.mean([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15])#复杂度之一
            '''
            '''t = np.mean([v1,v2,v3,v4])
            t1 = math.pow((v1-t),2);t2 = math.pow((v2-t),2);t3 = math.pow((v3-t),2);t4 = math.pow((v4-t),2);
            ave_t = np.mean([t1, t2, t3, t4])  # 复杂度之一,聚类特征？
            '''
            # av = np.mean([v1, v2, v3, v4])
            # pe = abs(x-av)

            datas0 = np.array(
                [v2, v3, v4, w1, w2, w3, w4, w5, w6, w7, w8, wL, D])  # 像素周围的12个像素存到datas0,D是复杂度，先添加进去，后续删除
            x0 = np.mean([v2, v3, v4, w1, w2, w3, w4, w5, w6, w7, w8, wL])  # 12像素的平均值，x0为论文中的m
            if np.mod(r + c, 2) == 0 and flag == 0:  # 偶数点
                label.append([D, x0]);
                data.append(datas0)  # 此时的label存放的就是聚类的特征向量；data存储的为像素周围的12个像素
            if np.mod(r + c, 2) == 1 and flag == 1:  # 奇数点
                label.append([D, x0]);
                data.append(datas0)

    # data=np.array(data);data=data.astype('float32')/255.;label=np.array(label);label=label.astype('float32')/255  #数据转换成np数组归一化，提高计算效率
    return data, label


# ------------------------------------------establish the neural networks------------------------------------------
def predictor_NN(data, label):  # 此时的label存放的就是聚类的特征向量(c,m)；data存储的为像素周围的12个像素
    EYI = 12;
    EY1 = 64;
    EY2 = 64;
    EYO = 18;
    BS = 2000;
    EP = 40  # EY1=64为输出节点个数；EYO=18为类的数量，即分成多少类别;EYI=12，12个像素
    inputs = Input(shape=(EYI,))  # 确定输入形状EYI,形状为一行12列，12个像素
    hidden_layer_1 = Dense(EY1, activation='relu')(inputs)
    hidden_layer_1 = Dropout(rate=0.01)(hidden_layer_1)
    hidden_layer_2 = Dense(EY2, activation='relu')(hidden_layer_1)
    hidden_layer_2 = Dropout(rate=0.01)(hidden_layer_2)
    hidden_layer_3 = Dense(EY1, activation='relu')(hidden_layer_2)
    hidden_layer_3 = Dropout(rate=0.01)(hidden_layer_3)
    hidden_layer_4 = Dense(EY2, activation='relu')(hidden_layer_3)
    hidden_layer_4 = Dropout(rate=0.01)(hidden_layer_4)
    hidden_layer_5 = Dense(EY1, activation='relu')(hidden_layer_4)
    hidden_layer_5 = Dropout(rate=0.01)(hidden_layer_5)
    hidden_layer_6 = Dense(EY2, activation='relu')(hidden_layer_5)
    hidden_layer_6 = Dropout(rate=0.01)(hidden_layer_6)
    Output = Dense(EYO, activation='relu')(hidden_layer_6)
    PNN = Model(inputs=inputs, outputs=Output)
    # print('=',PNN)
    print('summary:',PNN.summary())
    PNN.compile(optimizer='adam',
                loss='mse')  # the loss is computed through mean square error, and the optimzer is adam
    PNN.fit(data, label, epochs=EP, batch_size=BS, shuffle=True,
            verbose=2)  # shuffle=True是否打乱输入样本顺序；verbose=2 ，日志显示，0无，1进度条，2每个迭代

    return PNN


# ---------------------------------------here is the function for k-means-------------------------------------------
def clustering(data, k):
    # here we apply k-means on the data after densing
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 100000000, 0.000000001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)  # 这里的label已经把像素附上标签了，是一维数组，为size(data)*1
    label = np.array(label);
    classes = []  # this list is used to store the x-y information of each class
    frequencies = []  # this list is used to record the number of pixels in each class
    color = ['r', 'gold', 'k', 'orange', 'gray', 'lightcoral', 'seashell', 'bisque', 'tan', 'g', 'b', 'm', 'pink',
             'purple', 'c', 'y', 'sage']  # colorbar
    # plt.subplot(121)
    # plot the scatter diagram after k-means
    for i in range(k):
        cl = data[label.ravel() == i]  # label.ravel()将多维ndarray降维一维等效于flatten
        classes.append(cl)
        plt.scatter(classes[i][:, 0], classes[i][:, 1], cmap='rainbow')
        frequencies.append(len(classes[i][:, 0]))

    plt.xlabel('Character-1:Complexity level'), plt.ylabel('Character-2:Mean value')  # ;plt.grid(ls='--')
    plt.scatter(center[:, 0], center[:, 1], c='indigo', marker='s')  # plot the center of each class)
    # plt.show()  #主动注释，立flag
    x = range(k)

    # reassign the label by the codes below
    f = sorted(frequencies);
    f.reverse();
    print('f', f)
    l = {};
    ls = {}
    for i in range(k):
        l[i] = frequencies[i]
        ls[f[i]] = i
    label_trans = {};
    c = []
    for i in range(max(label.shape)):
        var_0 = l[label[i][0]]  # 根据label查到对应频数
        var_1 = ls[var_0]  # 根据频数查到新的组别
        c.append(var_1)
        label[i][0] = var_1
    ct = Counter(c);
    y = []
    x = sorted(ct.keys())
    for i in x:
        y.append(ct[i])
    y = np.array(y) / 10000
    name_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
    # plt.subplot(122)
    # plt.xlabel('Clusters');plt.ylabel('Population($\mathregular{10^4}$)')  #主动注释，立flag
    # plt.grid(ls='--')

    # plt.bar(x,y,tick_label=name_list);plt.show() #主动注释，立flag

    return label


# ----------------------------------------------read the data from the file----------------------------------------------------
def read_data(flag):
    if flag == 0:
        ### making the training data set
        # img0=cv2.imread('C:/Users/11748/Desktop\DNN_Code\DNN_RDH\Lena.bmp',0);img0_0,img00=img_division(img0,0);img0_1,img01=img_division(img0,1)
        # img1=cv2.imread('C:/Users/11748/Desktop\DNN_Code\DNN_RDH\lake.bmp',0);img1_0,img10=img_division(img1,0);img1_1,img11=img_division(img1,1)
        # test_data=np.vstack((img1_0,img1_1,img0_0,img0_1));test_label=np.vstack((img10,img11,img00,img01))

        img0 = cv2.imread('./Kodak24gray512bmp/kodim01.bmp', 0);
        img0_0, img_00 = img_division(img0, 0);
        img0_1, img_01 = img_division(img0, 1)
        img1 = cv2.imread('./Kodak24gray512bmp/kodim02.bmp', 0);
        img1_0, img_10 = img_division(img1, 0);
        img1_1, img_11 = img_division(img1, 1)
        img2 = cv2.imread('./Kodak24gray512bmp/kodim03.bmp', 0);
        img2_0, img_20 = img_division(img2, 0);
        img2_1, img_21 = img_division(img2, 1)
        img3 = cv2.imread('./Kodak24gray512bmp/kodim04.bmp', 0);
        img3_0, img_30 = img_division(img3, 0);
        img3_1, img_31 = img_division(img3, 1)
        img4 = cv2.imread('./Kodak24gray512bmp/kodim05.bmp', 0);
        img4_0, img_40 = img_division(img4, 0);
        img4_1, img_41 = img_division(img4, 1)
        img5 = cv2.imread('./Kodak24gray512bmp/kodim06.bmp', 0);
        img5_0, img_50 = img_division(img5, 0);
        img5_1, img_51 = img_division(img5, 1)
        img6 = cv2.imread('./Kodak24gray512bmp/kodim07.bmp', 0);
        img6_0, img_60 = img_division(img6, 0);
        img6_1, img_61 = img_division(img6, 1)
        img7 = cv2.imread('./Kodak24gray512bmp/kodim08.bmp', 0);
        img7_0, img_70 = img_division(img7, 0);
        img7_1, img_71 = img_division(img7, 1)
        img8 = cv2.imread('./Kodak24gray512bmp/kodim09.bmp', 0);
        img8_0, img_80 = img_division(img8, 0);
        img8_1, img_81 = img_division(img8, 1)
        img9 = cv2.imread('./Kodak24gray512bmp/kodim10.bmp', 0);
        img9_0, img_90 = img_division(img9, 0);
        img9_1, img_91 = img_division(img9, 1)
        img10 = cv2.imread('./Kodak24gray512bmp/kodim11.bmp', 0);
        img10_0, img_100 = img_division(img10, 0);
        img10_1, img_101 = img_division(img10, 1)
        img11 = cv2.imread('./Kodak24gray512bmp/kodim12.bmp', 0);
        img11_0, img_110 = img_division(img11, 0);
        img11_1, img_111 = img_division(img11, 1)
        img12 = cv2.imread('./Kodak24gray512bmp/kodim13.bmp', 0);
        img12_0, img_120 = img_division(img12, 0);
        img12_1, img_121 = img_division(img12, 1)
        img13 = cv2.imread('./Kodak24gray512bmp/kodim14.bmp', 0);
        img13_0, img_130 = img_division(img13, 0);
        img13_1, img_131 = img_division(img13, 1)
        img14 = cv2.imread('./Kodak24gray512bmp/kodim15.bmp', 0);
        img14_0, img_140 = img_division(img14, 0);
        img14_1, img_141 = img_division(img14, 1)
        img15 = cv2.imread('./Kodak24gray512bmp/kodim16.bmp', 0);
        img15_0, img_150 = img_division(img15, 0);
        img15_1, img_151 = img_division(img15, 1)
        img16 = cv2.imread('./Kodak24gray512bmp/kodim17.bmp', 0);
        img16_0, img_160 = img_division(img16, 0);
        img16_1, img_161 = img_division(img16, 1)
        img17 = cv2.imread('./Kodak24gray512bmp/kodim18.bmp', 0);
        img17_0, img_170 = img_division(img17, 0);
        img17_1, img_171 = img_division(img17, 1)
        img18 = cv2.imread('./Kodak24gray512bmp/kodim19.bmp', 0);
        img18_0, img_180 = img_division(img18, 0);
        img18_1, img_181 = img_division(img18, 1)
        img19 = cv2.imread('./Kodak24gray512bmp/kodim20.bmp', 0);
        img19_0, img_190 = img_division(img19, 0);
        img19_1, img_191 = img_division(img19, 1)
        img20 = cv2.imread('./Kodak24gray512bmp/kodim21.bmp', 0);
        img20_0, img_200 = img_division(img20, 0);
        img20_1, img_201 = img_division(img20, 1)
        img21 = cv2.imread('./Kodak24gray512bmp/kodim22.bmp', 0);
        img21_0, img_210 = img_division(img21, 0);
        img21_1, img_211 = img_division(img21, 1)
        img22 = cv2.imread('./Kodak24gray512bmp/kodim23.bmp', 0);
        img22_0, img_220 = img_division(img22, 0);
        img22_1, img_221 = img_division(img22, 1)
        img23 = cv2.imread('./Kodak24gray512bmp/kodim24.bmp', 0);
        img23_0, img_230 = img_division(img23, 0);
        img23_1, img_231 = img_division(img23, 1)

        '''img0=cv2.imread('C:/Users/11748/Desktop\Kodak24gray512bmp\Lena.bmp',0);img0_0,img_00=img_division(img0,0);img0_1,img_01=img_division(img0,1)
        img1=cv2.imread('C:/Users/11748/Desktop\Kodak24gray512bmp\Airplane.bmp',0);img1_0,img_10=img_division(img1,0);img1_1,img_11=img_division(img1,1)
        img2=cv2.imread('C:/Users/11748/Desktop\Kodak24gray512bmp\kodim03.bmp',0);img2_0,img_20=img_division(img2,0);img2_1,img_21=img_division(img2,1)
        img3=cv2.imread('C:/Users/11748/Desktop\Kodak24gray512bmp\Elaine.bmp',0);img3_0,img_30=img_division(img3,0);img3_1,img_31=img_division(img3,1)
        img4=cv2.imread('C:/Users/11748/Desktop\Kodak24gray512bmp\Lake.bmp',0);img4_0,img_40=img_division(img4,0);img4_1,img_41=img_division(img4,1)
        img5=cv2.imread('C:/Users/11748/Desktop\Kodak24gray512bmp\Boat.bmp',0);img5_0,img_50=img_division(img5,0);img5_1,img_51=img_division(img5,1)
        '''

        '''
        np.vstack
        按垂直方向（行顺序）堆叠数组构成一个新的数组
        堆叠的数组需要具有相同的维度

        '''
        test_data = np.vstack((img0_0, img0_1, img1_0, img1_1, img2_0, img2_1, img3_0, img3_1, img4_0, img4_1, img5_0,
                               img5_1, img6_0, img6_1, img7_0, img7_1, img8_0, img8_1, img9_0, img9_1, img10_0, img10_1,
                               img11_0, img11_1, img12_0, img12_1, img13_0, img13_1, img14_0, img14_1, img15_0, img15_1,
                               img16_0, img16_1, img17_0, img17_1, img18_0, img18_1, img19_0, img19_1, img20_0, img20_1,
                               img21_0, img21_1, img22_0, img22_1, img23_0, img23_1))
        # print(img_10,img_11,img_00,img_01,img_20,img_21,img_30,img_31,img_40,img_41,img_50,img_51,img_60,img_61,img_70,img_71,img_80,img_81,img_90,img_91,img_100,img_101,img_110,img_111,img_120,img_121,img_130,img_131,img_140,img_141,img_150,img_151,img_160,img_161,img_170,img_171,img_180,img_181,img_190,img_191,img_200,img_201,img_210,img_211,img_220,img_221,img_230,img_231)

        test_label = np.vstack((img_00, img_01, img_10, img_11, img_20, img_21, img_30, img_31, img_40, img_41, img_50,
                                img_51, img_60, img_61, img_70, img_71, img_80, img_81, img_90, img_91, img_100,
                                img_101, img_110, img_111, img_120, img_121, img_130, img_131, img_140, img_141,
                                img_150, img_151, img_160, img_161, img_170, img_171, img_180, img_181, img_190,
                                img_191, img_200, img_201, img_210, img_211, img_220, img_221, img_230, img_231))
        # test_data = np.vstack((img0_0,img0_1,img1_0,img1_1,img2_0,img2_1,img3_0,img3_1,img4_0,img4_1,img5_0,img5_1))

        # print(img_10,img_11,img_00,img_01,img_20,img_21,img_30,img_31,img_40,img_41,img_50,img_51,img_60,img_61,img_70,img_71,img_80,img_81,img_90,img_91,img_100,img_101,img_110,img_111,img_120,img_121,img_130,img_131,img_140,img_141,img_150,img_151,img_160,img_161,img_170,img_171,img_180,img_181,img_190,img_191,img_200,img_201,img_210,img_211,img_220,img_221,img_230,img_231)

        # test_label = np.vstack((img_00,img_01,img_10,img_11,img_20,img_21,img_30,img_31,img_40,img_41, img_50,img_51))

        # test_data = np.vstack((img0_0, img0_1))
        # test_label = np.vstack((img_00, img_01))
        print('test_data_shape', test_data.shape)
        print('test_data', test_data)
        print('test_label_shape', test_label.shape)
        print('test_label', test_label)
        row, col = test_data.shape
        # print('row', row)
        label1 = []
        data1 = []
        for r in range(0, row):
            # print(test_data[r][0][2])

            if test_data[r][12] < 350:  # 小于复杂度阈值独立摘出来 第13列是D
                a = test_data[r].tolist()  # 把满足小于350阈值的行以列表的形式赋值给a

                data1.append(a)  # 存放到data1

                # print('data2 ', test_data)
            if test_label[r][0] < 350:  # 把对应的特征向量独立摘出来，即（背景像素平均值和复杂度D）
                b = test_label[r].tolist()  # 把满足小于350阈值的行以列表的形式赋值给b
                label1.append(b)

        # print('data1', data1)
        print('len(data1)', len(data1))
        for i in range(0, len(data1)):
            data1[i].pop(12)  # 删除第13列的D，恢复12列训练像素
        # for i in range(0, len(label1)):
        # label1[i].pop(0)
        # print('data1', data1)
        data1 = np.array(data1)
        label1 = np.array(label1)
        print('data1.shape', data1.shape)
        print('data1', data1)

        # print(type(data1))
        # print(type(label1))
        print('label1.shape', label1.shape)
        print('label1', label1)

        data1 = np.array(data1);
        data1 = data1.astype('float32') / 255.;
        label1 = np.array(label1);
        label1 = label1.astype('float32') / 255  # 数据转换成np数组归一化，提高计算效率
        # test_data=np.vstack((img1_0,img1_1,img0_0,img0_1,img2_0,img2_1))
        # test_label=np.vstack((img10,img11,img00,img01,img20,img21))
        return data1, label1  ##test_data是12个像素，四个数组是落在一起排列的，test_label是特征向量

    else:
        img = cv2.imread('./DNN_Code/DNN_RDH/Lena.bmp', 0)
        # img=cv2.imread('E:/2019SeniorYearInBG/data_sets/koadak24/gray.tiff',0)
        return img


if __name__ == '__main__':

    user = input('Do you want to train? ');
    k = 18;
    file_name = 'blank.txt'
    if user.upper() == '100':  # build a new model
        train_data, train_label = read_data(
            flag=0)  # 读取图像。制作训练集，train_data,train_label 分别是所有每个像素周围的12个像素的行摞在一起的数组，label是特征集（c，m）
        print('train_data.shape', train_data.shape)
        print('train_data', train_data)
        print('train_label.shape', train_label.shape)
        print('train_label', train_label)
        train_label = clustering(train_label, k)  # 聚类，label是特征集(c，m),k是聚类数目
        print('train_label1.shape', train_label.shape)
        print('train_label1', train_label)
        # scio.savemat('C:/Users/11748/Desktop/model/train_label.mat', {'train_label': train_label})  # 保存矩阵D命名为Dlena.mat，key命名为Dbis
        labels = np.zeros((max(train_label.shape), k)) + 0.01
        # print('labels = ',labels)
        for i in range(max(train_label.shape)):
            # print('train_label[i]',train_label[i])
            labels[i][train_label[i]] = 0.99
        # print('secondlabels',labels)
        # scio.savemat('C:/Users/11748/Desktop/model/labels.mat', {'labels': labels})  # 保存矩阵D命名为Dlena.mat，key命名为Dbis

        # PNN.save('Pnn1.h5')
        PNN = predivt_nn(train_data, labels)
        # PNN.save('C:/Users/11748/Desktop/model/Pnn5.h5')
        PNN.save(PNN, "pnn.pth")
    else:
        PNN = load_model('./DNN_RDH/DNN.h5')  # Load the original model
