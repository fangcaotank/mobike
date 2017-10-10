# encoding: utf-8
'''
Created on 2017年6月30日

@author: c
'''
from mytools import *
import csv
import collections
"""
第一个模型是把 userid 和 Start、time 结合  （第一重要）
第二个模型是把Start 和 time结合  （当 userid不存在时为第一重要性）
第三个模型是把 Start 泛化             （寻找最近的，然后使用模型2）
"""


tr=csv.DictReader(open('trainfeature_v1.csv'))
te=csv.DictReader(open('testfeature.csv'))

user_habit_dict=dict()

start_end_dict=dict()
i=0
for rec in tr:
    print(i)
    i+=1
    user=rec['userid']
    start=rec['geohashed_start_loc']
    end=rec['geohashed_end_loc']
    hour=int(rec['hour'])
    if user_habit_dict.__contains__(user):
        user_habit_dict[user].append([start,end,hour])
    else:
        user_habit_dict[user]=[[start,end,hour]]
    if start_end_dict.__contains__(start):
        start_end_dict[start].append(end)
    else:
        start_end_dict[start]=[end]

print('comp1')
sub=open('submission.csv','w')

for rec in te:
    user=rec['userid']
    start=rec['geohashed_start_loc']
    hour=int(rec['hour'])
    result=[]
    if user_habit_dict.__contains__(user):
        level1=set()
        level2=set()
        level3=set()
        level4=set()
        tup=user_habit_dict[user]
        for item in tup:
            rec_start,rec_end,rec_hour=item
            if rec_start==start and rec_hour==hour:
                level1.add(rec_end)
            elif loc_2_dis(rec_start,start)<0.2 and rec_hour==hour:
                level2.add(rec_end)
            elif loc_2_dis(rec_start,start)<0.2 and abs(rec_hour-hour)<2:
                level3.add(rec_end)
            else:
                level4.add(rec_end)
        result.extend(level1)
        result.extend(level2)
        result.extend(level3)
        result.extend(level4)
    elif start_end_dict.__contains__(start):
        tup=start_end_dict[start]
        a=collections.Counter(tup)
        b=dict(sorted(a.items(), key=lambda d:d[1], reverse = True))
        result.extend(b.keys()) 
    result.append('123')
    result.append('223')
    result.append('323')
    string=rec['orderid']
    for item in result:
        string+=','+item
    sub.write(string+'\n')
    print(string)
print('comp2')



