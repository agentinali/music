import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import csv
import cv2
import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import pandas as pd

csv_file='train_truth.csv'
test_file='test_truth.csv'
train_path='./data/music_train/'
test_path='./data/music_test/'


def read_csv(file_name,img_path):
    # 開啟 CSV 檔案
    img_data = []
    img_dat = []
    img_lab = []
    img_name=[]
    with open(file_name) as csvfile:
    # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)
        for i in rows:
            if i[1] == 'category': continue
            img_lab.append(np.array(int(i[1])))
            lena = cv2.imread(img_path + i[0]).astype(np.float32) / 255

            img_data.append(torch.from_numpy(lena).reshape(-1,394,520))
            # img_data.append(torch.from_numpy(lena.transpose(2, 0, 1)))
            # img_data.append(torch.from_numpy(lena))

        # res = pd.DataFrame({'name': img_name, 'label': img_lab, 'data': img_dat})
        # res.to_csv('img_data.csv', index=False)
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
        layer1.add_module('conv1', nn.Conv2d(3, 256, 5, 1))
        # layer1.add_module('nb1', nn.BatchNorm2d(256))
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1=layer1

        layer2=nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(256,128,5,1))
        # layer2.add_module('nb2', nn.BatchNorm2d(128))
        layer2.add_module('relu2', nn.ReLU())
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        # layer3 = nn.Sequential()
        # layer3.add_module('conv3', nn.Conv2d(128, 128, 1, 1))
        # layer3.add_module('nb3', nn.BatchNorm2d(128))
        # layer3.add_module('relu3', nn.ReLU())
        # layer3.add_module('pool3', nn.MaxPool2d(2, 2))
        # self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('conv4', nn.Conv2d(128, 64, 5, 1))
        # layer4.add_module('nb4', nn.BatchNorm2d(64))
        layer4.add_module('relu4', nn.ReLU())
        layer4.add_module('pool4', nn.MaxPool2d(2, 2))
        self.layer4 = layer4

        layer9 = nn.Sequential()
        layer9.add_module('fc1', nn.Linear(175680,1024))
        layer9.add_module('dp1', nn.Dropout())
        layer9.add_module('fc1_relu', nn.ReLU(True))
        layer9.add_module('fc2', nn.Linear(1024, num_classes))

        self.layer9 = layer9

    def forward(self, x):
        conv1=self.layer1(x)
        conv2=self.layer2(conv1)
        conv4=self.layer4(conv2)
        # conv4 = self.layer4(conv3)
        fc_input=conv4.view(conv4.size(0),-1)
        fc_out=self.layer9(fc_input)

        return fc_out

if __name__=='__main__':

    batch_size = 8
    num_workers = 0
    num_epoches = 2
    learning_rate = 0.01
    num_classes = 88
    weight_decay = 0.1
    k=5
    net = My_Net()
    print(net)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    tr_datas,tr_labels = read_csv(csv_file,img_path=train_path)
    tx_datas, tx_labels = read_csv(test_file, img_path=test_path)
    # test_dataset = Mydataset(tx_datas,tx_labels)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    ## K折訓練
    fold_size = len(tr_datas) // k
    for i in range(k):
        val_data = tr_datas[i * fold_size:(i + 1) * fold_size]
        val_targets = tr_labels[i * fold_size:(i + 1) * fold_size]
        partial_train_data = tr_datas[:i * fold_size] + tr_datas[(i + 1) * fold_size:]
        partial_train_targets = tr_labels[:i * fold_size] + tr_labels[(i + 1) * fold_size:]
        print(str(i) +'K折訓練')
        tr_dataset = Mydataset(partial_train_data, partial_train_targets)
        val_dataset = Mydataset(val_data, val_targets)

        train_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
        net = My_Net()
        if  torch.cuda.is_available():
            net = My_Net().cuda()  ### 实例化模型
        optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # 將batch_size設為1 可看單張圖片
    # images,labels=dataiter.next()
    # images=images.numpy()

    # plt.imshow(np.squeeze(images))  # 顯示圖片
    # plt.axis('off')  # 不顯示座標軸
    # plt.show()
    # print('type(x): ', type(x))
    # print('x.dtype: ', x.dtype)  # x的具体类型
        print('-' * 25, '第', i + 1, '折', '-' * 25)
# 开始训练
        for epoch in range(num_epoches):
            # print('epoch {}'.format(epoch + 1))
            # print('*' * 10)
            running_loss = 0.0
            running_acc = 0.0
            iter_no=len(train_loader)
            for i, data in enumerate(train_loader, 1):
                x,y = data
                img = Variable(x)
                label = Variable(y)
                if torch.cuda.is_available():
                    img = Variable(x).cuda()
                    label = Variable(y).cuda()
            # 向前传播
                out = net(img)
            #
            # print(out.dtype)
            # print(label.dtype)
            # print(len(out),len(label))
                loss = loss_func(out, label.long())
                # running_loss += loss.item() * label.size(0)
                # _, pred = torch.max(out, 1)
                # num_correct = (pred == label.long()).sum()
                # accuracy = (pred == label.long()).float().mean()
                # running_acc += num_correct.item()
        # 向后传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(i)
                if (i+1) % 10 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.6f' %(epoch + 1, num_epoches, i+1,iter_no,loss.item()))
                    # print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, num_epoches, running_loss / (batch_size * i),running_acc / (batch_size * i)))
            # print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, running_loss / (len(train_loader)), running_acc / (len(train_loader))))
            net.eval()
            with torch.no_grad():
                eval_loss = 0
                eval_acc = 0
                for data in val_loader:
                    x, y = data
                    img = Variable(x)
                    label = Variable(y)
                    if torch.cuda.is_available():
                        img = Variable(x).cuda()
                        label = Variable(y).cuda()

                    out = net(img)
                    loss = loss_func(out, label.long())
                    eval_loss += loss.item()
                    # eval_loss += loss.item() * label.size(0)
                    _, pred = torch.max(out, 1)
                    num_correct = (pred == label.long()).sum()
                    eval_acc += num_correct.item()
                print('eVal Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_loader)), eval_acc / (len(val_loader))))
                print()
            net.train()

# 保存模型
    torch.save(net.state_dict(), './cnn.pth')

    net.eval()
    with torch.no_grad():  # disable auto-grad
        # # 將真正測試集餵給model 做預測
        #     x = Variable(test_tensor).cuda()
        #      y = Variable(y).cuda()
        #     test_pred = model(x).cuda()

        if torch.cuda.is_available():
            test_pred = net(Variable(test_tensor).cuda())
        else:
            test_pred = net(test_tensor)

        #     labelout = torch.max(test_pred, 1)[1].data.numpy()
        labelout = torch.argmax(test_pred, dim=1)
        outputs = labelout.cpu()
        print(outputs)
        # #將預測結果轉成pandas 資料格式並寫入 CSV檔，最後將此結果上傳Kaggle
        res = pd.DataFrame({'samp_id': range(1, 7173), 'label': outputs})
        res.to_csv('test_sub.csv', index=False)