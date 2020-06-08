import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
# from torchvision import transforms
import csv
import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
# from keras.utils import np_utils


csv_file='train_truth.csv'
test_file='test_truth.csv'
train_path='./data/music_train/'
test_path='./data/music_test/'


def read_csv(file_name,img_path):
    # 開啟 CSV 檔案
    img_data = []
    img_lab = []
    with open(file_name) as csvfile:
    # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)
        for i in rows:
            if i[1] == 'category': continue
            img_lab.append(np.array(int(i[1])))
            lena = mpimg.imread(img_path + i[0])
            # img_data.append(lena.transpose(2,0,1))
            img_data.append(torch.from_numpy(lena.transpose(2, 0, 1)))
    return img_data, img_lab

def read_img(img_names,labes,img_path):

    data_summy=[]
    lab_dat=[]
    for i, lab in zip(img_names, labes):
        lena = mpimg.imread(img_path + i)  # 讀取和程式碼處於同一目錄下的 lena.png# 此時 lena 就已經是一個 np.array 了，可以對它進行任意處理
        print(lena.shape)  # (512, 512, 3)
        print(len(lena))
        # data_summy.append(lena.reshape(4,394,520))
        data_summy.append(lena.transpose(2, 0, 1))
        lab_dat.append(lab)
    # plt.imshow(lena)  # 顯示圖片
    # plt.axis('off')  # 不顯示座標軸
    # plt.show()
    return data_summy,lab_dat

class Mydataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return  len(self.data)
    def __getitem__(self, item):
        data=self.data[item]
        label=self.label[item]
        # label = torch.tensor(self.label[item])
        return data, label


class My_Net(nn.Module):
    def __init__(self):
        super(My_Net, self).__init__()
        layer1=nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(4, 32, 3, 2, padding=1))
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1=layer1

        layer2=nn.Sequential()
        layer2.add_module('conv2',nn.Conv2d(32,64,3,2,padding=1))
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(64, 128, 3, 2, padding=1))
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(6144,512))
        layer4.add_module('fc1_relu', nn.ReLU(True))
        layer4.add_module('fc2', nn.Linear(512, 256))
        layer4.add_module('fc2_relu', nn.ReLU(True))
        layer4.add_module('fc3', nn.Linear(256, num_classes))
        self.layer4 = layer4


    def forward(self, x):
        conv1=self.layer1(x)
        conv2=self.layer2(conv1)
        conv3=self.layer3(conv2)
        fc_input=conv3.view(conv3.size(0),-1)
        fc_out=self.layer4(fc_input)

        return fc_out


def train(net, train_features, train_labels, num_epochs, learning_rate, weight_decay, batch_size , fg):
    train_ls, test_ls = [], []  ##存储train_loss,test_loss
    dataset = Mydataset(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)
    ptc=len(train_features)
    ### 将数据封装成 Dataloder 对应步骤（2）

    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:  ###分批训练
            output = net(X)
            loss = loss_func(output, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ### 得到每个epoch的 loss 和 accuracy
        # train_ls.append(log_rmse(0, net, train_features, train_labels))
        if fg:
            test_ls.append(log_rmse(1, net, train_features, train_labels,ptc))
        else:
            train_ls.append(log_rmse(0, net, train_features, train_labels,ptc))
    # print(train_ls,test_ls)
    return test_ls if fg else train_ls


def log_rmse(flag, net, x, y,ptc):
    if flag == 1:  ### valid 数据集
        net.eval()
    dataset = Mydataset(x, y)
    train_iter = DataLoader(dataset, ptc, shuffle=True)
    for x, y in train_iter:  ###分批训练
        output = net(x)
        result = torch.max(output, 1)[1].view(y.size())
        corrects = (result.data == y.data).sum().item()
        accuracy = corrects * 100.0 / len(y)  #### 5 是 batch_size
        loss = loss_func(output, y)
    net.train()

    return (loss.data.item(), accuracy)


if __name__=='__main__':
    # model = My_Net()
    # print(model)
    batch_size = 32
    num_workers = 0
    num_epoches = 20
    learning_rate = 0.001
    num_classes = 88
    weight_decay = 0.1
    k=10
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # criterion = nn.CrossEntropyLoss()
    loss_func = nn.CrossEntropyLoss()
    tr_datas,tr_labels = read_csv(csv_file,img_path=train_path)
    tx_datas, tx_labels = read_csv(test_file, img_path=test_path)

    tr_dataset = Mydataset(tr_datas,tr_labels)
    tx_dataset = Mydataset(tx_datas, tx_labels)

    train_loader=DataLoader(dataset=tr_dataset,batch_size=batch_size)
    test_loader = DataLoader(dataset=tx_dataset, batch_size=batch_size)
    # dataiter=iter(train_data)

    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    fold_size = len(tr_datas) // k
    for i in range(k):
        val_data = tr_datas[i * fold_size:(i + 1) * fold_size]
        val_targets = tr_labels[i * fold_size:(i + 1) * fold_size]
        partial_train_data = tr_datas[:i * fold_size] + tr_datas[(i + 1) * fold_size:]
        partial_train_targets = tr_labels[:i * fold_size] + tr_labels[(i + 1) * fold_size:]

        net = My_Net()  ### 实例化模型
            ### 每份数据进行训练,体现步骤三####
        train_ls = train(net, partial_train_data,partial_train_targets, num_epoches , learning_rate, weight_decay, batch_size,fg=True)
        valid_ls = train(net, val_data, val_targets, num_epoches, learning_rate, weight_decay, batch_size , fg=False)

        print('*' * 25, '第', i + 1, '折', '*' * 25)
        print('train_loss:%.6f' % train_ls[-1][0], 'train_acc:%.4f\n' % valid_ls[-1][1], \
                  'valid loss:%.6f' % valid_ls[-1][0], 'valid_acc:%.4f' % valid_ls[-1][1])
        train_loss_sum += train_ls[-1][0]
        valid_loss_sum += valid_ls[-1][0]
        train_acc_sum += train_ls[-1][1]
        valid_acc_sum += valid_ls[-1][1]
    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
        ####体现步骤四#####
    print('train_loss_sum:%.4f' % (train_loss_sum / k), 'train_acc_sum:%.4f\n' % (train_acc_sum / k), \
              'valid_loss_sum:%.4f' % (valid_loss_sum / k), 'valid_acc_sum:%.4f' % (valid_acc_sum / k))


    # 將batch_size設為1 可看單張圖片
    # images,labels=dataiter.next()
    # images=images.numpy()
    input()
    # plt.imshow(np.squeeze(images))  # 顯示圖片
    # plt.axis('off')  # 不顯示座標軸
    # plt.show()
    # print('type(x): ', type(x))
    # print('x.dtype: ', x.dtype)  # x的具体类型












# 开始训练
    for epoch in range(num_epoches):
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):
            x,y = data
            img = Variable(x)
            label = Variable(y)
            # 向前传播
            out = model(img)
            #
            # print(out.dtype)
            # print(label.dtype)
            # print(len(out),len(label))
            loss = criterion(out, label.long())
            # print(loss)

            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label.long()).sum()

            accuracy = (pred == label.long()).float().mean()
            running_acc += num_correct.item()
        # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(i)
            if i % 30 == 0:
                print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, num_epoches, running_loss / (batch_size * i),running_acc / (batch_size * i)))
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, running_loss / (len(train_loader)), running_acc / (len(train_loader))))
        model.eval()
        with torch.no_grad():
            eval_loss = 0
            eval_acc = 0
            for data in test_loader:
                img, label = data
                out = model(img)
                loss = criterion(out, label.long())
                eval_loss += loss.item() * label.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == label.long()).sum()
                eval_acc += num_correct.item()
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_loader)), eval_acc / (len(test_loader))))
            print()

# 保存模型
    torch.save(model.state_dict(), './cnn.pth')