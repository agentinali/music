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

batch_size = 32
num_workers = 0
num_epoches = 20
learning_rate = 1e-2
num_classes = 88


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
        layer1.add_module('conv1', nn.Conv2d(4, 32, 3, 1, padding=1))
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1=layer1

        layer2=nn.Sequential()
        layer2.add_module('conv2',nn.Conv2d(32,64,3,1,padding=1))
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(64, 128, 3, 1, padding=1))
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(407680,512))
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

if __name__=='__main__':
    model = My_Net()
    # print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    tr_datas,tr_labels = read_csv(csv_file,img_path=train_path)
    tx_datas, tx_labels = read_csv(test_file, img_path=test_path)

    tr_dataset = Mydataset(tr_datas,tr_labels)
    tx_dataset = Mydataset(tx_datas, tx_labels)

    train_loader=DataLoader(dataset=tr_dataset,batch_size=batch_size)
    test_loader = DataLoader(dataset=tx_dataset, batch_size=batch_size)
    # dataiter=iter(train_data)

    # 將batch_size設為1 可看單張圖片
    # images,labels=dataiter.next()
    # images=images.numpy()
    #
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
            num_correct = (pred == label).sum()

            accuracy = (pred == label).float().mean()
            running_acc += num_correct.item()
        # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(i)
            if i % 300 == 0:
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
                num_correct = (pred == label).sum()
                eval_acc += num_correct.item()
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_loader)), eval_acc / (len(test_loader))))
            print()

# 保存模型
    torch.save(model.state_dict(), './cnn.pth')