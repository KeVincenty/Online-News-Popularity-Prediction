import os
import csv

train_url_list = []
train_share_list = []
test_share_list = []

with open('/home/yinyuan/workspace/Online-News-Popularity-Prediction/train.csv','r') as f:
    i = 0
    data = csv.reader(f)
    for line in data:
        i+=1
        if i > 1:
            train_url_list.append(line[0])
            train_share_list.append(int(line[-1]))

with open('/home/yinyuan/workspace/Online-News-Popularity-Prediction/OnlineNewsPopularity.csv','r') as ff:
    ii = 0
    data_t = csv.reader(ff)
    for line_t in data_t:
        ii+=1
        if ii > 1:
            if line_t[0] not in train_url_list:
                # print(ii)
                test_share_list.append(int(line_t[-1]))
