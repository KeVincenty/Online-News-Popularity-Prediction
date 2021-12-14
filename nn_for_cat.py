import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

IS_EMBEDDING = True
IS_SCALE = True
IS_GROUP = True
EMBEDDING_DIM_CHANNEL = 20
EMBEDDING_DIM_DAY = 20
EMBEDDING_DIM_WEEKEND = 20
TRAIN_PATH = '/home/yinyuan/workspace/Online-News-Popularity-Prediction/train_27k.csv'
VAL_PATH = '/home/yinyuan/workspace/Online-News-Popularity-Prediction/val_3k.csv'
TEST_PATH = '/home/yinyuan/workspace/Online-News-Popularity-Prediction/test.csv'
TRAIN_BATCH_SIZE = 9000
VAL_BATCH_SIZE = 3000
TEST_BATCH_SIZE = 9644
TRAIN_EPOCH = 100
LINEAR_HIDDEN_DIM = 200
GRU_HIDDEN_DIM = 200
GRU_LAYERS = 2
CONV_NUM_FILTER = 10
CONV_FILTER_SIZE = 3
LR = 1e-2
MOMENTUM = 0.9


class NewsDataset(Dataset):
    def __init__(self, data_path, is_embedding=False, is_test=False):
        super().__init__()
        self.data_path = data_path
        self.is_test = is_test
        self.is_embedding = is_embedding
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        data_list = list(self.data[index].values())

        if not self.is_test:
            data = torch.Tensor(data_list[:-1])
            label = torch.Tensor([self.data[index]['shares']])
        else:
            data = torch.Tensor(data_list)
            label = torch.Tensor([0.])

        return data,label

    def read_raw_data(self):
        i = 0
        with open(self.data_path,'r') as f:
            data = csv.reader(f)
            for line in data:
                i+=1
                if i == 1:
                    keys = line
                if i>1:
                    if not self.is_test:
                        line_dict={}
                        for idx,key in enumerate(keys):
                            if key != 'url' and key != 'timedelta':
                                line_dict[key] = float(line[idx])
                            else:
                                continue
                        self.data.append(line_dict)
                    else:
                        self.data.append({key:float(line[index]) for index,key in enumerate(keys)})

def embed_data(data):
    embedding_channel = nn.Embedding(7, EMBEDDING_DIM_CHANNEL)
    embedding_day = nn.Embedding(7, EMBEDDING_DIM_DAY)
    embedding_weekend = nn.Embedding(2, EMBEDDING_DIM_WEEKEND)
    if sum(data[11:17]) != 0.:
        embedding_c = embedding_channel(torch.LongTensor([data[11:17].index(1.)]))
    else:
        embedding_c = embedding_channel(torch.LongTensor([6]))
    embedding_d = embedding_day(torch.LongTensor([data[29:36].index(1.)]))
    embedding_w = embedding_weekend(torch.LongTensor([int(data[36])]))
    data = torch.cat([torch.Tensor(data[0:11]).unsqueeze(0), embedding_c, torch.Tensor(data[17:29]).unsqueeze(0), embedding_d, embedding_w, torch.Tensor(data[37:-1]).unsqueeze(0)], dim=-1).squeeze(0)


def scale_data(data, is_embedding=False):
    mean = torch.Tensor([10.3987, 546.5147, 0.5482, 0.9965, 0.6892, 10.8837, 3.2936, 4.5441, 1.2499, 4.5482, 7.2238, 0.0529, 0.1780, 0.1579, 0.0586, 0.1853, 0.2126, 26.1068, 1153.9517, 312.3670, 13612.3541, 752324.0667, 259281.9381, 1117.1466, 5657.2112, 3135.8586, 3998.7554, 10329.2127, 6401.6976, 0.1680, 0.1864, 0.1875, 0.1833, 0.1438, 0.0619, 0.0690, 0.1309, 0.1846, 0.1413, 0.2163, 0.2238, 0.2340, 0.4434, 0.1193, 0.0396, 0.0166, 0.6822, 0.2879, 0.3538, 0.0954, 0.7567, -0.2595, -0.5219, -0.1075, 0.2824, 0.0714, 0.3418, 0.1561]).cuda()
    sd = torch.Tensor([2.1140, 471.1016, 3.5207, 5.2312, 3.2648, 11.3319, 3.8551, 8.3093, 4.1078, 0.8444, 1.9091, 0.2239, 0.3825, 0.3646, 0.2349, 0.3885, 0.4091, 69.6323, 3857.9422, 620.7761, 57985.2980, 214499.4242, 135100.5433, 1137.4426, 6098.7950, 1318.1338, 19738.4216, 41027.0592, 24211.0269, 0.3739, 0.3894, 0.3903, 0.3869, 0.3509, 0.2409, 0.2535, 0.3373, 0.2630, 0.2197, 0.2821, 0.2952, 0.2892, 0.1167, 0.0969, 0.0174, 0.0108, 0.1902, 0.1562, 0.1045, 0.0713, 0.2478, 0.1277, 0.2903, 0.0954, 0.3242, 0.2654, 0.1888, 0.2263]).cuda()
    data_m = (data-mean)/sd
    # if not is_embedding:
    data[:,0:11] = data_m[:,0:11]
    data[:,17:29] = data_m[:,17:29]
    data[:,37:] = data_m[:,37:]
    # else:
    #     data[0:11] = data_m[0:11]
    #     data[11+EMBEDDING_DIM_CHANNEL:29+EMBEDDING_DIM_CHANNEL] = data_m[17:29]
    #     data[37+EMBEDDING_DIM_CHANNEL+EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND:] = data_m[37:]
    return data

Trainset = NewsDataset(TRAIN_PATH, is_embedding=IS_EMBEDDING, is_test=False)
Trainset.read_raw_data()
Valset = NewsDataset(VAL_PATH, is_embedding=IS_EMBEDDING, is_test=False)
Valset.read_raw_data()
Testset = NewsDataset(TEST_PATH, is_embedding=IS_EMBEDDING, is_test=True)
Testset.read_raw_data()

TrainData = DataLoader(Trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
ValData = DataLoader(Valset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
TestData = DataLoader(Testset, batch_size=TEST_BATCH_SIZE, shuffle=False)

class LinearNet(nn.Module):
    def __init__(self, hidden_dim, is_embedding):
        super().__init__()
        self.is_embedding = is_embedding
        if is_embedding:
            self.input_feature = 44+EMBEDDING_DIM_CHANNEL+EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND
            self.embedding_channel = nn.Embedding(7, EMBEDDING_DIM_CHANNEL)
            self.embedding_day = nn.Embedding(7, EMBEDDING_DIM_DAY)
            self.embedding_weekend = nn.Embedding(2, EMBEDDING_DIM_WEEKEND)
        else:
            self.input_feature = 58
        self.linear_1 = nn.Linear(self.input_feature, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.linear_3 = nn.Linear(2*hidden_dim, 4*hidden_dim)
        self.linear_4 = nn.Linear(4*hidden_dim, 2*hidden_dim)
        self.linear_5 = nn.Linear(2*hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        if self.is_embedding:
            idx = x[:,11:17].nonzero()
            embed_idx = torch.full((x.size(0),1),6).cuda()
            embed_idx[idx[:,0]] = idx[:,1].unsqueeze(-1)
            embedding_c = self.embedding_channel(embed_idx).squeeze(1)
            embedding_d = self.embedding_day(x[:,29:36].nonzero()[:,1])
            embedding_w = self.embedding_weekend(x[:,36].long())
            x = torch.cat([x[:,0:11], embedding_c, x[:,17:29], embedding_d, embedding_w, x[:,37:]], dim=-1)
        x = self.linear_1(x)
        x = self.dropout(self.activation(x))
        x = self.linear_2(x)
        x = self.dropout(self.activation(x))
        x = self.linear_3(x)
        x = self.dropout(self.activation(x))
        x = self.linear_4(x)
        x = self.dropout(self.activation(x))
        x = self.linear_5(x)
        return nn.ReLU()(x)

class GRUNet(nn.Module):
    def __init__(self, hidden_dim, num_layers, is_embedding, is_group):
        super().__init__()
        self.is_embedding = is_embedding
        self.is_group = is_group
        if is_embedding:
            self.input_feature = 44+EMBEDDING_DIM_CHANNEL+EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND
            self.embedding_channel = nn.Embedding(7, EMBEDDING_DIM_CHANNEL)
            self.embedding_day = nn.Embedding(7, EMBEDDING_DIM_DAY)
            self.embedding_weekend = nn.Embedding(2, EMBEDDING_DIM_WEEKEND)
        else:
            self.input_feature = 58
        if self.is_group:
            self.input_dim = 100
            self.linear_words = nn.Linear(6, self.input_dim)
            self.linear_links = nn.Linear(5, self.input_dim)
            self.linear_media = nn.Linear(2, self.input_dim)
            self.linear_time = nn.Linear(EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND, self.input_dim) if self.is_embedding else nn.Linear(8, self.input_dim)
            self.linear_keywords = nn.Linear(10+EMBEDDING_DIM_CHANNEL, self.input_dim) if self.is_embedding else nn.Linear(16, self.input_dim) 
            self.linear_nlp = nn.Linear(21, self.input_dim)
        else:
            self.input_dim = 1
        self.gru = nn.GRU(self.input_dim, hidden_dim, num_layers=num_layers, dropout=0.2)
        self.linear_1 = nn.Linear(6*hidden_dim, hidden_dim) if self.is_group else nn.Linear(hidden_dim*self.input_feature, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        if self.is_embedding:
            idx = x[:,11:17].nonzero()
            embed_idx = torch.full((x.size(0),1),6).cuda()
            embed_idx[idx[:,0]] = idx[:,1].unsqueeze(-1)
            embedding_c = self.embedding_channel(embed_idx).squeeze(1)
            embedding_d = self.embedding_day(x[:,29:36].nonzero()[:,1])
            embedding_w = self.embedding_weekend(x[:,36].long())
            x = torch.cat([x[:,0:11], embedding_c, x[:,17:29], embedding_d, embedding_w, x[:,37:]], dim=-1)
            # after embedding: [0:11] [11:11+EMBEDDING_DIM_CHANNEL] [11+EMBEDDING_DIM_CHANNEL:23+EMBEDDING_DIM_CHANNEL] [23+EMBEDDING_DIM_CHANNEL:23+EMBEDDING_DIM_CHANNEL+EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND] [23+EMBEDDING_DIM_CHANNEL+EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND:]
        if self.is_group:
            words_i = torch.cat([x[:,0:5], x[:,9].unsqueeze(-1)], dim=-1)
            links_i = torch.cat([x[:,5:7], x[:,20+EMBEDDING_DIM_CHANNEL:23+EMBEDDING_DIM_CHANNEL]], dim=-1) if self.is_embedding else torch.cat([x[:,5:7], x[:,26:29]], dim=-1)
            media_i = x[:,7:9]
            time_i = torch.cat([embedding_d, embedding_w], dim=-1) if self.is_embedding else x[:,29:37] 
            keywords_i = torch.cat([x[:,10].unsqueeze(-1), embedding_c, x[:,11+EMBEDDING_DIM_CHANNEL:20+EMBEDDING_DIM_CHANNEL]], dim=-1) if self.is_embedding else x[:,10:26] 
            nlp_i = x[:,23+EMBEDDING_DIM_CHANNEL+EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND:] if self.is_embedding else x[:,37:]

            words_v = self.linear_words(words_i).unsqueeze(0)
            links_v = self.linear_links(links_i).unsqueeze(0)
            media_v = self.linear_media(media_i).unsqueeze(0)
            time_v = self.linear_time(time_i).unsqueeze(0)
            keywords_v = self.linear_keywords(keywords_i).unsqueeze(0)
            nlp_v = self.linear_nlp(nlp_i).unsqueeze(0)
            input = torch.cat([words_v, links_v, media_v, time_v, keywords_v, nlp_v], dim=0)
        else:
            input = x.T.unsqueeze(-1)
        input = self.gru(input)
        input = input[0].transpose(0, 1)
        input = nn.Flatten()(input)
        input = self.linear_1(input)
        input = self.activation(input)
        input = self.linear_2(input)
        return nn.ReLU()(input)

class ConvNet(nn.Module):
    def __init__(self, num_filters, filter_size, is_embedding, is_group):
        super().__init__()
        self.is_embedding = is_embedding
        self.is_group = is_group
        if is_embedding:
            self.input_feature = 44+EMBEDDING_DIM_CHANNEL+EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND
            self.embedding_channel = nn.Embedding(7, EMBEDDING_DIM_CHANNEL)
            self.embedding_day = nn.Embedding(7, EMBEDDING_DIM_DAY)
            self.embedding_weekend = nn.Embedding(2, EMBEDDING_DIM_WEEKEND)
        else:
            self.input_feature = 58
        if self.is_group:
            self.input_dim = 100
            self.input_channel = 6
            self.linear_words = nn.Linear(6, self.input_dim)
            self.linear_links = nn.Linear(5, self.input_dim)
            self.linear_media = nn.Linear(2, self.input_dim)
            self.linear_time = nn.Linear(EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND, self.input_dim) if self.is_embedding else nn.Linear(8, self.input_dim)
            self.linear_keywords = nn.Linear(10+EMBEDDING_DIM_CHANNEL, self.input_dim) if self.is_embedding else nn.Linear(16, self.input_dim) 
            self.linear_nlp = nn.Linear(21, self.input_dim)
        else:
            self.input_channel = 1
            self.input_dim = 1
        self.conv_1 = nn.Conv1d(self.input_channel, num_filters, filter_size, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(self.input_dim//2) if self.is_group else nn.AdaptiveAvgPool1d(self.input_feature//2) 
        self.conv_2 = nn.Conv1d(num_filters, num_filters, filter_size, padding=1)
        self.linear_1 = nn.Linear(self.input_dim//2*num_filters, 100) if self.is_group else nn.Linear(self.input_feature//2*num_filters, 100)
        self.linear_2 = nn.Linear(100, 1)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if self.is_embedding:
            idx = x[:,11:17].nonzero()
            embed_idx = torch.full((x.size(0),1),6).cuda()
            embed_idx[idx[:,0]] = idx[:,1].unsqueeze(-1)
            embedding_c = self.embedding_channel(embed_idx).squeeze(1)
            embedding_d = self.embedding_day(x[:,29:36].nonzero()[:,1])
            embedding_w = self.embedding_weekend(x[:,36].long())
            x = torch.cat([x[:,0:11], embedding_c, x[:,17:29], embedding_d, embedding_w, x[:,37:]], dim=-1)
        if self.is_group:
            words_i = torch.cat([x[:,0:5], x[:,9].unsqueeze(-1)], dim=-1)
            links_i = torch.cat([x[:,5:7], x[:,20+EMBEDDING_DIM_CHANNEL:23+EMBEDDING_DIM_CHANNEL]], dim=-1) if self.is_embedding else torch.cat([x[:,5:7], x[:,26:29]], dim=-1)
            media_i = x[:,7:9]
            time_i = torch.cat([embedding_d, embedding_w], dim=-1) if self.is_embedding else x[:,29:37] 
            keywords_i = torch.cat([x[:,10].unsqueeze(-1), embedding_c, x[:,11+EMBEDDING_DIM_CHANNEL:20+EMBEDDING_DIM_CHANNEL]], dim=-1) if self.is_embedding else x[:,10:26] 
            nlp_i = x[:,23+EMBEDDING_DIM_CHANNEL+EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND:] if self.is_embedding else x[:,37:]

            words_v = self.linear_words(words_i).unsqueeze(1)
            links_v = self.linear_links(links_i).unsqueeze(1)
            media_v = self.linear_media(media_i).unsqueeze(1)
            time_v = self.linear_time(time_i).unsqueeze(1)
            keywords_v = self.linear_keywords(keywords_i).unsqueeze(1)
            nlp_v = self.linear_nlp(nlp_i).unsqueeze(1)
            input = torch.cat([words_v, links_v, media_v, time_v, keywords_v, nlp_v], dim=1)
        else:
            input = x.unsqueeze(1)

        input = self.conv_1(input)
        input = self.avgpool(input)
        input = self.conv_2(input)
        input = nn.Flatten()(input)
        input = self.activation(input)
        input = self.linear_1(input)
        input = self.activation(self.dropout(input))
        input = self.linear_2(input)
        return nn.ReLU()(input)

class ConvGRUNet(nn.Module):
    def __init__(self, num_filters, filter_size, hidden_dim, num_layers, is_embedding, is_group):
        super().__init__()
        self.is_embedding = is_embedding
        self.is_group = is_group
        if is_embedding:
            self.input_feature = 44+EMBEDDING_DIM_CHANNEL+EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND
            self.embedding_channel = nn.Embedding(7, EMBEDDING_DIM_CHANNEL)
            self.embedding_day = nn.Embedding(7, EMBEDDING_DIM_DAY)
            self.embedding_weekend = nn.Embedding(2, EMBEDDING_DIM_WEEKEND)
        else:
            self.input_feature = 58
        if self.is_group:
            self.input_dim = 100
            self.input_channel = 6
            self.linear_words = nn.Linear(6, self.input_dim)
            self.linear_links = nn.Linear(5, self.input_dim)
            self.linear_media = nn.Linear(2, self.input_dim)
            self.linear_time = nn.Linear(EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND, self.input_dim) if self.is_embedding else nn.Linear(8, self.input_dim)
            self.linear_keywords = nn.Linear(10+EMBEDDING_DIM_CHANNEL, self.input_dim) if self.is_embedding else nn.Linear(16, self.input_dim) 
            self.linear_nlp = nn.Linear(21, self.input_dim)
        else:
            self.input_channel = 1
            self.input_dim = 1
        self.conv_1 = nn.Conv1d(self.input_channel, num_filters, filter_size, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(self.input_dim//2) if self.is_group else nn.AdaptiveAvgPool1d(self.input_feature//2) 
        self.gru = nn.GRU(self.input_dim//2, hidden_dim, num_layers=num_layers, dropout=0.2)
        self.linear_1 = nn.Linear(num_filters*hidden_dim, 200)
        self.linear_2 = nn.Linear(200, 1)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        if self.is_embedding:
            idx = x[:,11:17].nonzero()
            embed_idx = torch.full((x.size(0),1),6).cuda()
            embed_idx[idx[:,0]] = idx[:,1].unsqueeze(-1)
            embedding_c = self.embedding_channel(embed_idx).squeeze(1)
            embedding_d = self.embedding_day(x[:,29:36].nonzero()[:,1])
            embedding_w = self.embedding_weekend(x[:,36].long())
            x = torch.cat([x[:,0:11], embedding_c, x[:,17:29], embedding_d, embedding_w, x[:,37:]], dim=-1)
        if self.is_group:
            words_i = torch.cat([x[:,0:5], x[:,9].unsqueeze(-1)], dim=-1)
            links_i = torch.cat([x[:,5:7], x[:,20+EMBEDDING_DIM_CHANNEL:23+EMBEDDING_DIM_CHANNEL]], dim=-1) if self.is_embedding else torch.cat([x[:,5:7], x[:,26:29]], dim=-1)
            media_i = x[:,7:9]
            time_i = torch.cat([embedding_d, embedding_w], dim=-1) if self.is_embedding else x[:,29:37] 
            keywords_i = torch.cat([x[:,10].unsqueeze(-1), embedding_c, x[:,11+EMBEDDING_DIM_CHANNEL:20+EMBEDDING_DIM_CHANNEL]], dim=-1) if self.is_embedding else x[:,10:26] 
            nlp_i = x[:,23+EMBEDDING_DIM_CHANNEL+EMBEDDING_DIM_DAY+EMBEDDING_DIM_WEEKEND:] if self.is_embedding else x[:,37:]

            words_v = self.linear_words(words_i).unsqueeze(1)
            links_v = self.linear_links(links_i).unsqueeze(1)
            media_v = self.linear_media(media_i).unsqueeze(1)
            time_v = self.linear_time(time_i).unsqueeze(1)
            keywords_v = self.linear_keywords(keywords_i).unsqueeze(1)
            nlp_v = self.linear_nlp(nlp_i).unsqueeze(1)
            input = torch.cat([words_v, links_v, media_v, time_v, keywords_v, nlp_v], dim=1)
        else:
            input = x.unsqueeze(1)

        input = self.conv_1(input)
        input = self.avgpool(input)
        input = input.transpose(0,1)
        input = self.gru(input)
        input = input[0].transpose(0, 1)
        input = nn.Flatten()(input)
        input = self.linear_1(input)
        input = self.activation(input)
        input = self.linear_2(input)
        return nn.ReLU()(input)


def train(traindata, valdata, model, optimizer, train_criterion, test_criterion, epochs):
    val_loss_list = []
    for epoch in range(epochs):
        for step, data in enumerate(traindata):
            inputs, label = data[0].cuda(), data[1].cuda()
            if IS_SCALE:
                inputs = scale_data(inputs)
            optimizer.zero_grad()
            output = model(inputs)
            loss = train_criterion(output, label)
            loss.backward()
            optimizer.step()
            print('Train | epoch:', epoch, 'step:', step, 'loss:', loss.item())
        for val_data in valdata:
            with torch.no_grad():
                val_inputs, val_label = val_data[0].cuda(), val_data[1].cuda()
                if IS_SCALE:
                    val_inputs = scale_data(val_inputs)
                val_pred = model(val_inputs)
                val_loss = test_criterion(val_pred, val_label)
                val_loss_list.append(val_loss.item())
                print('Val | epoch:', epoch, 'loss:', val_loss.item(), 'average_loss:', np.mean(val_loss_list))

    return model

# def ensemble_train(traindata, valdata, models, optimizer, train_criterion, test_criterion, epochs):
#     val_loss_list = []
#     output_list = []
#     val_pred_list = []
#     assert(isinstance(models, list))
#     for epoch in range(epochs):
#         for step, data in enumerate(traindata):
#             inputs, label = data[0].cuda(), data[1].cuda()
#             if IS_SCALE:
#                 inputs = scale_data(inputs)
#             optimizer.zero_grad()
#             for model in models:
#                 output = model(inputs)
#                 loss = train_criterion(output, label)
#                 loss.backward()
#                 optimizer.step()
#             print('Train | epoch:', epoch, 'step:', step, 'loss:', loss.item())
#         for val_data in valdata:
#             with torch.no_grad():
#                 val_inputs, val_label = val_data[0].cuda(), val_data[1].cuda()
#                 if IS_SCALE:
#                     val_inputs = scale_data(val_inputs)
#                 for model in models:
#                     val_pred_list.append(model(val_inputs))
#                 val_pred = sum(val_pred_list)/2
#                 val_loss = test_criterion(val_pred, val_label)
#                 val_loss_list.append(val_loss.item())
#                 print('Val | epoch:', epoch, 'loss:', val_loss.item(), 'average_loss:', np.mean(val_loss_list))

#     return models

@torch.no_grad()
def test(testdata, model):
    for data, _ in testdata:
        inputs = data.cuda()
        if IS_SCALE:
            inputs = scale_data(inputs)
        prediction = model(inputs)
        pred = prediction.cpu().numpy()
        np.savetxt('/home/yinyuan/workspace/Online-News-Popularity-Prediction/results_convgru_val_1.txt', pred)

# @torch.no_grad()
# def ensemble_test(testdata, model):
#     for data, _ in testdata:
#         inputs = data.cuda()
#         if IS_SCALE:
#             inputs = scale_data(inputs)
#         prediction = model(inputs)
#         pred = prediction.cpu().numpy()
#         np.savetxt('/home/yinyuan/workspace/Online-News-Popularity-Prediction/results_conv1d_val_5.txt', pred)


# model = LinearNet(LINEAR_HIDDEN_DIM, IS_EMBEDDING).cuda()
# model = GRUNet(GRU_HIDDEN_DIM, GRU_LAYERS, IS_EMBEDDING, IS_GROUP).cuda()
# model = ConvNet(CONV_NUM_FILTER, CONV_FILTER_SIZE, IS_EMBEDDING, IS_GROUP).cuda()
model = ConvGRUNet(20, 3, 100, 2, IS_EMBEDDING, IS_GROUP).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum = MOMENTUM)
train_criterion = nn.L1Loss()
test_criterion = nn.L1Loss()

model = train(TrainData, ValData, model, optimizer, train_criterion, test_criterion, TRAIN_EPOCH)
test(TestData, model)

# model_1 = GRUNet(GRU_HIDDEN_DIM, GRU_LAYERS, IS_EMBEDDING, IS_GROUP).cuda()
# model_2 = ConvNet(CONV_NUM_FILTER, CONV_FILTER_SIZE, IS_EMBEDDING, IS_GROUP).cuda()
# optimizer = torch.optim.Adam([{'params':model_1.parameters(), 'lr':1e-3}, 
#                             {'params':model_2.parameters()}], lr=LR)
# models = [model_1, model_2]
# train_criterion = nn.L1Loss()
# test_criterion = nn.L1Loss()

# models = ensemble_train(TrainData, ValData, models, optimizer, train_criterion, test_criterion, TRAIN_EPOCH)

