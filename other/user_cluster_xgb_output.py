# coding: utf-8

import os
import gc
import time
import Geohash
import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
import math
import mzgeohash
from collections import defaultdict

warnings.filterwarnings("ignore")

cache_path = 'mobike_cache_filter/'
train_path = 'data/trainall.csv'
test_path  = 'data/sub.csv'

########### setting ###########
aim = 4
pre_extract = 1
count = 1
load = False
predict = True
eval_ = True


use_days_to_train = 6
days_to_predict = 7
sample_thread = 7

    
if not os.path.exists(cache_path):
    os.mkdir(cache_path)

################## 预处理 #######################
t0 = time.time()
if load == True:
    # 时间串处理和leak特征提取
    train = pd.read_csv(train_path)
    train['starttime'] = pd.to_datetime(train['starttime'])
    train['date'] = pd.DatetimeIndex(train.starttime).date
    train['day'] =  pd.DatetimeIndex(train.starttime).day
    train['hour'] =  pd.DatetimeIndex(train.starttime).hour
    train['dayofweek'] = pd.DatetimeIndex(train.starttime).dayofweek

    test = pd.read_csv(test_path)
    test['starttime'] = pd.to_datetime(test['starttime'])
    test['date'] = pd.DatetimeIndex(test.starttime).date
    test['day'] =  pd.DatetimeIndex(test.starttime).day
    test['hour'] =  pd.DatetimeIndex(test.starttime).hour
    test['dayofweek'] = pd.DatetimeIndex(test.starttime).dayofweek
    test['geohashed_end_loc'] = np.nan
    test = test[train.columns]

    train_all = pd.concat([train, test], axis=0)#拼接以后，所有的数据
    train_all_2week = train_all[train_all.day>=17]#17号以后的数据
	
    del train, test; gc.collect()
    
################## 辅助函数 #####################
def performance(f):                                                  #定义装饰器函数，功能是传进来的函数进行包装并返回包装后的函数  
    def fn(*args, **kw):                                             #对传进来的函数进行包装的函数  
        t_start = time.time()                                        #记录函数开始时间   
        r = f(*args, **kw)                                           #调用函数  
        t_end = time.time()                                          #记录函数结束时间   
        print ('call %s() in %fs' % (f.__name__, (t_end - t_start))) #打印调用函数的属性信息，并打印调用函数所用的时间  
        return r                                                     #返回包装后的函数      
    return fn                                                        #调用包装后的函数  

def dump_feature(f):                                                  #定义装饰器函数，功能是传进来的函数进行包装并返回包装后的函数  
    def fn(*args, **kw):                                             #对传进来的函数进行包装的函数  
        t_start = time.time()
        if len(args) == 0:
            dump_path = cache_path + f.__name__+ 'null' +'.pickle'
        else:
            dump_path = cache_path + f.__name__+ str(list(args)) +'.pickle'
        if os.path.exists(dump_path):
            r = pd.read_pickle(dump_path)
        else:
            r = f(*args, **kw)                                           
            r.to_pickle(dump_path)
        gc.collect()
        t_end = time.time() 
        print ('call %s() in %fs' % (f.__name__, (t_end - t_start)))
        return r                                                          
    return fn    


# 分组排序
def rank(data, feat1, feat2, ascending, rank_name):
    # 这部分比自己的实现的要好非常多，好好学习，大概比我实现的快六十倍
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data[rank_name] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)[rank_name].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data[rank_name] = data[rank_name] - data['min_rank']
    del data['min_rank']
    return data

def rank3(data, feat1, feat2, feat3, ascending, rank_name):
    # 这部分比自己的实现的要好非常多，好好学习，大概比我实现的快六十倍
    data.sort_values([feat1, feat2, feat3], inplace=True, ascending=ascending)
    data[rank_name] = range(data.shape[0])
    min_rank = data.groupby([feat1, feat2],as_index=False)[rank_name].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=[feat1, feat2],how='left')
    data[rank_name] = data[rank_name] - data['min_rank']
    del data['min_rank']
    return data

# 对结果进行整理
def reshape(pred):
    result = pred
    result = rank(result,'orderid','pred', False, 'rank')
    result = result[result['rank']<3][['orderid','geohashed_end_loc','rank']]
    result = result.set_index(['orderid','rank']).unstack()
    result.reset_index(inplace=True)
    result['orderid'] = result['orderid'].astype('int')
    result.columns = ['orderid', 0, 1, 2]
    return result

# 测评函数
def measure(result):
    result_path = cache_path + 'true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        true = dict(zip(train['orderid'].values,train['geohashed_end_loc']))
        pickle.dump(true,open(result_path, 'wb+'))
    data = result.copy()
    data['true'] = data['orderid'].map(true)
    score = (sum(data['true']==data[0])
             +sum(data['true']==data[1])/2
             +sum(data['true']==data[2])/3)/data.shape[0]
    return score

# 获取真实标签
def get_label(result):
    dump_path = cache_path + 'true.pkl'
    if os.path.exists(dump_path):
        true = pd.read_pickle(dump_path)
    else:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        test['geohashed_end_loc'] = np.nan
        data = pd.concat([train,test])
        true = dict(zip(data['orderid'].values, data['geohashed_end_loc']))
        pd.to_pickle(true, dump_path)
    if len(result) != 0:   
        result['label'] = result['orderid'].map(true)
        result['label'] = (result['label'] == result['geohashed_end_loc']).astype('int')
    return result

############### 并行处理函数 ###################
from joblib import Parallel, delayed
import itertools
import gc

def execu(func):
    print (func)
    x = func[0](*func[1])
    del x; gc.collect()
    print (func, 'done')

def applyParallel_feature(funcs, execu, num_threads):
    ''' 利用joblib来并行提取特征
    '''
    with Parallel(n_jobs=num_threads) as parallel:
        retLst = parallel( delayed(execu)(func) for func in funcs )
        return None 

def applyParallel(dfGrouped, func, num_threads):
    '''利用joblib来生成特征样本
    '''
    with Parallel(n_jobs=num_threads) as parallel:
        retLst = parallel( delayed(func)(*group) for group in dfGrouped )
        return pd.concat(retLst) 

###########leak特征的计算###################
@dump_feature
def get_orderid_user_leak():
    train = train_all_2week.copy()
    
    train.sort_values(by=['userid', 'starttime'], inplace=True, ascending=True)
    train['last_userid'] = train.userid.shift(1)
    train['next_userid'] = train.userid.shift(-1)
    
    train['user_last_order_time_diff'] = (pd.DatetimeIndex(train.starttime) - pd.DatetimeIndex(train.starttime.shift(1))).total_seconds().values
    train['user_next_order_time_diff'] = (pd.DatetimeIndex(train.starttime.shift(-1)) - pd.DatetimeIndex(train.starttime)).total_seconds().values
    
    train['user_last_sloc'] = train.geohashed_start_loc.shift(1)
    train['user_next_sloc'] = train.geohashed_start_loc.shift(-1)
    
    train = train.loc[:, ['orderid', 'last_userid',  'next_userid', 'user_last_order_time_diff', 'user_next_order_time_diff', 'user_last_sloc', 'user_next_sloc']]
    return train
    
@dump_feature
def get_orderid_bike_leak():
    train = train_all_2week.copy()
    
    train.sort_values(by=['bikeid', 'starttime'], inplace=True, ascending=True)
    train['next_bikeid'] = train.bikeid.shift(-1)
    train['bike_next_order_time_diff'] = (pd.DatetimeIndex(train.starttime.shift(-1)) - pd.DatetimeIndex(train.starttime)).total_seconds().values
    train['bike_next_sloc'] = train.geohashed_start_loc.shift(-1)
    
    train = train.loc[:, ['orderid', 'next_bikeid',  'bike_next_order_time_diff', 'bike_next_sloc']]
    return train

#筛选起始地点去向最多的20个地点
@dump_feature
def get_clusterloc_to_loc(end_day, prediction_type):
    
    train = train_all[ train_all.day <= end_day-prediction_type ]
    location_pairs = train.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['geohashed_end_loc'].agg({'sloc_eloc_count':'count'})
    
    temp = train_all[train_all.day==end_day]
    location_pairs_count = defaultdict(lambda: defaultdict(lambda: 0))
    
    loc_neighbors = dict()
    
    print ('mzgeohash start')
    for x in  temp.geohashed_start_loc.drop_duplicates().values:
        loc_neighbors[x] = mzgeohash.neighbors(x).values()
    print ('mzgeohash end')
    
    for x in location_pairs.values:
        location_pairs_count[x[0]][x[1]] = x[2]
    
    cluster_pairs_count = defaultdict(lambda: defaultdict(lambda: 0))

    for x in temp.geohashed_start_loc.drop_duplicates().values:
        for loc in loc_neighbors[x]:
            for a, b in location_pairs_count[loc].iteritems():
                cluster_pairs_count[x][a] += b
    
    slocs, elocs, counts = [], [], []
    for sloc, elocs_count in cluster_pairs_count.iteritems():
        for eloc, count in elocs_count.iteritems():
            slocs.append(sloc)
            elocs.append(eloc)
            counts.append(count)
   
    location_pairs = pd.DataFrame({'geohashed_start_loc': slocs,  'geohashed_end_loc': elocs, 'sloc_eloc_count': counts})
    location_pairs.sort_values(by=['sloc_eloc_count'], inplace=True)
    eloc_sloc = location_pairs.groupby('geohashed_start_loc').tail(10)[['geohashed_start_loc',  'geohashed_end_loc']]
    
    result = pd.merge(temp[['orderid', 'geohashed_start_loc']], eloc_sloc, on='geohashed_start_loc', how='left')
    result = result[['orderid', 'geohashed_end_loc']].drop_duplicates()
    print ('loc_to_loc', result.shape)
    
    return result

def get_sample(end_day, prediction_type):
    dump_path = cache_path + 'sample_filter_%s_%s_27_.pickle' %(end_day, prediction_type)
    dump_path_ = cache_path + 'sample_filter_%s_%s_27_fan.pickle' %(end_day, prediction_type)
    if os.path.exists(dump_path_):
        samples = pd.read_pickle(dump_path_)
    else:
        
        samples = pd.read_pickle(dump_path)[['orderid', 'geohashed_end_loc']]
        loc_to_loc = get_clusterloc_to_loc(end_day, prediction_type) 

        sloc_bike_nextsloc =  get_orderid_bike_leak()
        sloc_bike_nextsloc = pd.merge(sloc_bike_nextsloc, train_all[['orderid', 'bikeid']], on='orderid', how='left')
        sloc_bike_nextsloc =  sloc_bike_nextsloc[sloc_bike_nextsloc.bikeid==sloc_bike_nextsloc.next_bikeid][['orderid','bike_next_sloc']]
        sloc_bike_nextsloc.columns = ['orderid', 'geohashed_end_loc']
        
        
        samples = pd.concat([samples, loc_to_loc, sloc_bike_nextsloc], axis=0).drop_duplicates()
        
        if end_day > 24:
            samples =  pd.concat([samples, sloc_bike_nextsloc], axis=0).drop_duplicates(keep=False)
        
        temp = train_all[ train_all.day==end_day ]
        temp.rename(columns={'geohashed_end_loc': 'label'}, inplace=True)
        samples = pd.merge(samples[samples.orderid.isin(temp.orderid)], temp, on='orderid', how='left')

        # 删除起始地点和目的地点相同的样本  和 异常值
        samples  = samples[samples['geohashed_end_loc'] != samples['geohashed_start_loc']]
        samples  = samples[(~samples['geohashed_end_loc'].isnull()) & (~samples['geohashed_start_loc'].isnull())]
        print (end_day, prediction_type, 'sample shape', samples.shape)
        samples.to_pickle(dump_path_)
    return samples

################## 特征提取 #####################

###############未交叉特征##################

# 获取用户的出发计数
@dump_feature
def get_user_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]
    user_count = train.groupby('userid',as_index=False)['geohashed_end_loc'].agg({'user_count':'count'})
    return user_count

@dump_feature
def get_user_count_(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]
    user_count = train.groupby('userid',as_index=False)['geohashed_end_loc'].agg({'user_count_':'count'})
    return user_count

# 获取自行车的出发计数，衡量车子的好坏或者车子是否很难找到
@dump_feature
def get_bike_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]
    bike_count = train.groupby('bikeid',as_index=False)['geohashed_end_loc'].agg({'bikeid_count':'count'})
    return bike_count

# 获取目标地点的热度(出发地地)
@dump_feature
def get_sloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    sloc_count = train.groupby('geohashed_start_loc', as_index=False)['userid'].agg({'sloc_count': 'count'})
    sloc_count.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
    return sloc_count

@dump_feature
def get_sloc_real_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    sloc_count = train.groupby('geohashed_start_loc', as_index=False)['userid'].agg({'sloc_real_count': 'count'})
    return sloc_count

# 获取目标地点的热度(目的地)
@dump_feature
def get_eloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    eloc_count = train.groupby('geohashed_end_loc', as_index=False)['userid'].agg({'eloc_count': 'count'})
    return eloc_count

@dump_feature
def get_hour_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    hour_count = train.groupby('hour', as_index=False)['userid'].agg({'hour_count': 'count'})
    return hour_count

@dump_feature
def get_biketype_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    biketype_count = train.groupby('biketype', as_index=False)['userid'].agg({'biketype_count': 'count'})
    return biketype_count

#####去重统计
# 获取自行车的去重热度
@dump_feature
def get_bike_unqiue_user_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = train[['userid', 'bikeid']].drop_duplicates()
    bike_user_count = train.groupby(['bikeid'], as_index=False)['userid'].agg({'bike_user_unique_count': 'count'})
    return bike_user_count

# 获取目的地的去重热度
@dump_feature
def get_eloc_unique_user_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = train[['geohashed_end_loc', 'userid']].drop_duplicates()
    eloc_user_count = train.groupby(['geohashed_end_loc'], as_index=False)['userid'].agg({'eloc_user_unique_count': 'count'})
    return eloc_user_count

# 获取出发地的去重热度
@dump_feature
def get_sloc_unique_user_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = train[['geohashed_start_loc', 'userid']].drop_duplicates()
    sloc_user_count = train.groupby(['geohashed_start_loc'], as_index=False)['userid'].agg({'sloc_user_unique_count': 'count'})
    sloc_user_count.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
    return sloc_user_count

#################二交叉特征#####################

# 获取用户从某个地点出发的行为次数
@dump_feature
def get_user_sloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]
    user_sloc_count = train.groupby(['userid','geohashed_start_loc'], as_index=False)['userid'].agg({'user_sloc_count': 'count'})
    user_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
    return user_sloc_count

# 获取用户从某个地点出发的行为比率
@dump_feature
def get_user_sloc_ratio(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]
    user_sloc_count = train.groupby(['userid','geohashed_start_loc'])['userid'].agg({'user_sloc_count_ratio': 'count'})
    user_count = train.groupby(['userid'])['userid'].agg({'user_sloc_count_ratio': 'count'})  
    user_sloc_count = user_sloc_count.div(user_count).reset_index()
    user_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
    return user_sloc_count

# 获取用户去过某个地点历史行为次数
@dump_feature
def get_user_eloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]    
    user_eloc_count = train.groupby(['userid','geohashed_end_loc'], as_index=False)['userid'].agg({'user_eloc_count': 'count'})
    return user_eloc_count

# 获取用户去过某个地点的行为比率
@dump_feature
def get_user_eloc_ratio(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]
    user_eloc_count = train.groupby(['userid','geohashed_end_loc'])['userid'].agg({'user_eloc_count_ratio': 'count'})
    user_count = train.groupby(['userid'])['userid'].agg({'user_eloc_count_ratio': 'count'})  
    user_eloc_count = user_eloc_count.div(user_count).reset_index()
    return user_eloc_count

# 路径走过次数
@dump_feature
def get_sloc_eloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    sloc_eloc_count = train.groupby(['geohashed_start_loc','geohashed_end_loc'], as_index=False)['userid'].agg({'sloc_eloc_count' :'count'})
    return sloc_eloc_count

# 路径走过次数
@dump_feature
def get_cluster_sloc_eloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    location_pairs = train.groupby(['geohashed_start_loc','geohashed_end_loc'], as_index=False)['userid'].agg({'sloc_eloc_count' :'count'})

    location_pairs_count = defaultdict(lambda: defaultdict(lambda: 0))
    
    loc_neighbors = dict()
    
    print ('mzgeohash start')
    for x in location_pairs.geohashed_start_loc.drop_duplicates().values:
        loc_neighbors[x] = mzgeohash.neighbors(x).values()
    print ('mzgeohash end')
    
    for x in location_pairs.values:
        location_pairs_count[x[0]][x[1]] = x[2]
    
    cluster_pairs_count = defaultdict(lambda: defaultdict(lambda: 0))
    for x in location_pairs.values:
        for loc in loc_neighbors[x[0]]:
            cluster_pairs_count[x[0]][x[1]] += location_pairs_count[loc][x[1]]
    
    slocs, elocs, counts = [], [], []
    for sloc, elocs_count in cluster_pairs_count.iteritems():
        for eloc, count in elocs_count.iteritems():
            slocs.append(sloc)
            elocs.append(eloc)
            counts.append(count)
   
    location_pairs = pd.DataFrame({'geohashed_start_loc': slocs,  'geohashed_end_loc': elocs, 'cluster_sloc_eloc_count': counts})
    
    return location_pairs

# 路径走过次数
@dump_feature
def get_sloc_eloc_ratio(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    sloc_eloc_count = train.groupby(['geohashed_start_loc','geohashed_end_loc'])['userid'].agg({'sloc_eloc_count_ratio' :'count'})
    sloc_count = train.groupby(['geohashed_start_loc'])['userid'].agg({'sloc_eloc_count_ratio' :'count'})
    sloc_eloc_count = sloc_eloc_count.div(sloc_count).reset_index()
    return sloc_eloc_count

@dump_feature
def get_cluster_sloc_eloc_ratio(start_day, end_day, label_day, prediction_type):
    sloc_eloc_count = get_cluster_sloc_eloc_count(start_day, end_day, label_day, prediction_type)
    
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    
    location_pairs = train.groupby(['geohashed_start_loc'], as_index=False)['userid'].agg({'user_count' :'count'})
    
    location_pairs_count = defaultdict(lambda: 0)
    
    loc_neighbors = dict()
    
    print ('mzgeohash start')
    for x in location_pairs.geohashed_start_loc.drop_duplicates().values:
        loc_neighbors[x] = mzgeohash.neighbors(x).values()
    print ('mzgeohash end')
    
    for x in location_pairs.values:
        location_pairs_count[x[0]] = x[1]
    
    cluster_pairs_count = defaultdict(lambda: 0)
    for x in location_pairs.values:
        for loc in loc_neighbors[x[0]]:
            cluster_pairs_count[x[0]] += location_pairs_count[loc]
    
    slocs,  counts = [], []
    for sloc, count in cluster_pairs_count.iteritems():
            slocs.append(sloc)
            counts.append(count)
   
    location_pairs = pd.DataFrame({'geohashed_start_loc': slocs, 'sloc_count': counts})
    
    sloc_eloc_count = pd.merge(sloc_eloc_count, location_pairs, on='geohashed_start_loc', how='left')
    
    sloc_eloc_count['cluster_sloc_eloc_count_ratio'] = sloc_eloc_count.cluster_sloc_eloc_count.values / sloc_eloc_count.sloc_count.values
    
    sloc_eloc_count = sloc_eloc_count[['geohashed_start_loc','geohashed_end_loc', 'cluster_sloc_eloc_count_ratio']]
    
    return sloc_eloc_count



# 路径折返次数
@dump_feature
def get_eloc_sloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    eloc_sloc_count = train.groupby(['geohashed_end_loc','geohashed_start_loc'], as_index=False)['userid'].agg({'eloc_sloc_count' :'count'})
    eloc_sloc_count.rename(columns = {'geohashed_start_loc':'geohashed_end_loc','geohashed_end_loc':'geohashed_start_loc'},inplace=True)
    return eloc_sloc_count

# 路径折返次数
@dump_feature
def get_eloc_sloc_ratio(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    eloc_sloc_count = train.groupby(['geohashed_end_loc','geohashed_start_loc'])['userid'].agg({'eloc_sloc_count_ratio' :'count'})
    eloc_count = train.groupby(['geohashed_end_loc'])['userid'].agg({'eloc_sloc_count_ratio' :'count'})
    eloc_sloc_count = eloc_sloc_count.div(eloc_count).reset_index()
    eloc_sloc_count.rename(columns = {'geohashed_start_loc':'geohashed_end_loc','geohashed_end_loc':'geohashed_start_loc'},inplace=True)
    return eloc_sloc_count

# 获取目标地点的当天的小时热度(出发地)
@dump_feature
def get_sloc_today_hour_count():
    sloc_hour_count = train_all.groupby(['geohashed_start_loc', 'date', 'hour'], as_index=False)['userid'].agg({'sloc_today_hour_count': 'count'})
    sloc_hour_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'}, inplace=True)
    return sloc_hour_count

# 获取目标地点的当天的小时热度(出发地)
@dump_feature
def get_sloc_today_hour_ratio():
    print ('get_sloc_today_hour_ratio type true')
    sloc_hour_count = train_all.groupby(['geohashed_start_loc', 'date_hour'])['userid'].agg({'sloc_today_hour_count_ratio': 'count'})
    sloc_count = train_all.groupby(['geohashed_start_loc'])['userid'].agg({'sloc_today_hour_count_ratio': 'count'})
    sloc_hour_count =  sloc_hour_count.div(sloc_count).reset_index()
    sloc_hour_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'}, inplace=True)
    return sloc_hour_count

# 获取目标地点的历史的小时热度(出发地)
@dump_feature
def get_sloc_hour_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ] 
    sloc_hour_count = train.groupby(['geohashed_start_loc', 'hour'], as_index=False)['userid'].agg({'sloc_hour_count': 'count'})
    sloc_hour_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'}, inplace=True)
    return sloc_hour_count

# 获取目标地点的历史的小时热度(出发地)
@dump_feature
def get_sloc_hour_ratio(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ] 
    sloc_hour_count = train.groupby(['geohashed_start_loc', 'hour'])['userid'].agg({'sloc_hour_count_ratio': 'count'})
    
    sloc_hour = train.groupby(['geohashed_start_loc'])['userid'].agg({'sloc_hour_count_ratio': 'count'})   
    sloc_hour_count = sloc_hour_count.div(sloc_hour).reset_index()
    
    sloc_hour_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'}, inplace=True)
    return sloc_hour_count

@dump_feature
def get_hour_sloc_ratio(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ] 
    hour_sloc_count = train.groupby([ 'hour', 'geohashed_start_loc'])['userid'].agg({'hour_sloc_count_ratio': 'count'})
    hour_count = train.groupby([ 'hour'])['userid'].agg({'hour_sloc_count_ratio': 'count'}) 
    hour_sloc_count = hour_sloc_count.div(hour_count).reset_index()
    hour_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'}, inplace=True)
    return hour_sloc_count

# 获取目标地点的历史小时热度(目的地)
@dump_feature
def get_eloc_hour_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    eloc_count = train.groupby(['geohashed_end_loc', 'hour'], as_index=False)['userid'].agg({'eloc_hour_count': 'count'})
    return eloc_count

# 获取目标地点的历史小时热度(目的地)
@dump_feature
def get_eloc_hour_ratio(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    eloc_count = train.groupby(['geohashed_end_loc', 'hour'])['userid'].agg({'eloc_hour_count_ratio': 'count'})
    eloc = train.groupby(['geohashed_end_loc'])['userid'].agg({'eloc_hour_count_ratio': 'count'})
    eloc_count = eloc_count.div(eloc).reset_index()
    return eloc_count

# 获取目标地点的历史小时热度(目的地)
@dump_feature
def get_hour_eloc_ratio(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    eloc_count = train.groupby(['hour', 'geohashed_end_loc'])['userid'].agg({'hour_eloc_count_ratio': 'count'})
    eloc = train.groupby(['hour'])['userid'].agg({'hour_eloc_count_ratio': 'count'})
    eloc_count = eloc_count.div(eloc).reset_index()
    return eloc_count

#################三交叉特征##################

# 获取用户从这个路径走过几次
@dump_feature
def get_user_sloc_eloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    user_sloc_eloc_count = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'], as_index=False)['userid'].agg({'user_sloc_eloc_count' :'count'})
    return user_sloc_eloc_count

@dump_feature
def get_user_sloc_eloc_count_ratio(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    user_sloc_eloc_count = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'], as_index=False)['userid'].agg({'user_sloc_eloc_count' :'count'})
    user_sloc_count = train.groupby(['userid','geohashed_start_loc'], as_index=False)['userid'].agg({'user_sloc_count' :'count'})
    user_sloc_eloc_count = pd.merge(user_sloc_eloc_count, user_sloc_count , on=['userid','geohashed_start_loc'], how='left')
    user_sloc_eloc_count['user_sloc_eloc_count_ratio'] = user_sloc_eloc_count.user_sloc_eloc_count.values / user_sloc_eloc_count.user_sloc_count.values
    user_sloc_eloc_count= user_sloc_eloc_count[['userid','geohashed_start_loc','geohashed_end_loc', 'user_sloc_eloc_count_ratio']]
    return user_sloc_eloc_count

@dump_feature
def get_user_cluster_sloc_eloc_count_ratio(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    
    user_sloc_eloc_count = get_user_cluster_sloc_eloc_count(start_day, end_day, label_day, prediction_type)
    
    location_pairs = train.groupby(['userid','geohashed_start_loc'], as_index=False)['userid'].agg({'user_sloc_count' :'count'})
    
    location_pairs_count = defaultdict(lambda: defaultdict(lambda: 0))
    
    loc_neighbors = dict()
    
    print ('mzgeohash start')
    for x in location_pairs.geohashed_start_loc.drop_duplicates().values:
        loc_neighbors[x] = mzgeohash.neighbors(x).values()
    print ('mzgeohash end')
    
    for x in location_pairs.values:
        location_pairs_count[x[0]][x[1]] = x[2]
    
    cluster_pairs_count = defaultdict(lambda: defaultdict(lambda: 0))
    for x in location_pairs.values:
        for loc in loc_neighbors[x[1]]:
            cluster_pairs_count[x[0]][x[1]] += location_pairs_count[x[0]][loc]
    
    userids, slocs,  counts = [], [], []
    for userid, slocs_counts in cluster_pairs_count.iteritems():
        for sloc, count in slocs_counts.iteritems():
            userids.append(userid)
            slocs.append(sloc)
            counts.append(count)
   
    location_pairs = pd.DataFrame({'userid': userids, 'geohashed_start_loc': slocs, 'user_sloc_count': counts})
    
    user_sloc_eloc_count = pd.merge(user_sloc_eloc_count, location_pairs , on=['userid', 'geohashed_start_loc'], how='left')
    
    user_sloc_eloc_count['cluster_user_sloc_eloc_coun_ratio'] = user_sloc_eloc_count.user_cluster_sloc_eloc_count.values / user_sloc_eloc_count.user_sloc_count.values
    
    user_sloc_eloc_count = user_sloc_eloc_count[['userid', 'geohashed_start_loc', 'geohashed_end_loc', 'cluster_user_sloc_eloc_coun_ratio']]
    
    return user_sloc_eloc_count


# 获取用户从这个路径走过几次
@dump_feature
def get_user_cluster_sloc_eloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    
    location_pairs = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'], as_index=False)['userid'].agg({'user_sloc_eloc_count' :'count'})
    
    location_pairs_count = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    
    loc_neighbors = dict()
    
    print ('mzgeohash start')
    for x in location_pairs.geohashed_start_loc.drop_duplicates().values:
        loc_neighbors[x] = mzgeohash.neighbors(x).values()
    print ('mzgeohash end')
    
    for x in location_pairs.values:
        location_pairs_count[x[0]][x[1]][x[2]] = x[3]
    
    cluster_pairs_count = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    for x in location_pairs.values:
        for loc in loc_neighbors[x[1]]:
            cluster_pairs_count[x[0]][x[1]][x[2]] += location_pairs_count[x[0]][loc][x[2]]
    
    userids, slocs, elocs, counts = [], [], [], []
    for userid, slocs_elocs_count in cluster_pairs_count.iteritems():
        for sloc, elocs_count in slocs_elocs_count.iteritems():
            for eloc, count in elocs_count.iteritems():
                userids.append(userid)
                slocs.append(sloc)
                elocs.append(eloc)
                counts.append(count)
   
    location_pairs = pd.DataFrame({'userid': userids ,'geohashed_start_loc': slocs,  'geohashed_end_loc': elocs, 'user_cluster_sloc_eloc_count': counts})
    
    return location_pairs

# 获取用户从这个路径折返过几次
@dump_feature
def get_user_eloc_sloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    user_eloc_sloc_count = train.groupby(['userid','geohashed_end_loc','geohashed_start_loc'], as_index=False)['userid'].agg({'user_eloc_sloc_count':'count'})
    user_eloc_sloc_count.rename(columns = {'geohashed_start_loc':'geohashed_end_loc','geohashed_end_loc':'geohashed_start_loc'},inplace=True)
    return user_eloc_sloc_count

# 获取用户在该小时去目的地的计数
@dump_feature
def get_user_hour_eloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    user_hour_eloc_count = train.groupby(['userid','hour','geohashed_end_loc'], as_index=False)['userid'].agg({'user_hour_eloc_count':'count'})
    return user_hour_eloc_count

# 获取用户在该小时从出发地的出发计数
@dump_feature
def get_user_hour_sloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]    
    user_hour_sloc_count = train.groupby(['userid','hour','geohashed_start_loc'], as_index=False)['userid'].agg({'user_hour_sloc_count':'count'})
    user_hour_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'}, inplace=True)
    return user_hour_sloc_count

# 获取该出发地在该小时去目的地的计数
@dump_feature
def get_sloc_hour_eloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    sloc_hour_eloc_count = train.groupby(['geohashed_start_loc','hour','geohashed_end_loc'],  as_index=False)['userid'].agg({'sloc_hour_eloc_count':'count'})
    return sloc_hour_eloc_count

# 获取该目的地在该小时去出发地的计数
@dump_feature
def get_eloc_hour_sloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    eloc_hour_sloc_count = train.groupby(['geohashed_end_loc','hour','geohashed_start_loc'], as_index=False)['userid'].agg({'eloc_hour_sloc_count':'count'})
    eloc_hour_sloc_count.rename(columns = {'geohashed_start_loc':'geohashed_end_loc','geohashed_end_loc':'geohashed_start_loc'},inplace=True)
    return eloc_hour_sloc_count

#################四交叉特征##################

# 获取用户在该小时从这个路径走过几次
@dump_feature
def get_user_hour_sloc_eloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    user_hour_sloc_eloc_count = train.groupby(['userid','hour','geohashed_start_loc','geohashed_end_loc'],                     as_index=False)['userid'].agg({'user_hour_sloc_eloc_count' :'count'})
    return user_hour_sloc_eloc_count


################### 全局角度的出发点统计 ############################

# 获取目标地点的热度(出发地地)
@dump_feature
def get_global_sloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all.copy() 
    sloc_count = train.groupby('geohashed_start_loc', as_index=False)['userid'].agg({'global_sloc_count': 'count'})
    sloc_count.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
    return sloc_count

# 获取出发地的去重热度
@dump_feature
def get_global_sloc_unique_user_count(start_day, end_day, label_day, prediction_type):
    train = train_all.copy()
    train = train[['geohashed_start_loc', 'userid']].drop_duplicates()
    sloc_user_count = train.groupby(['geohashed_start_loc'], as_index=False)['userid'].agg({'global_sloc_user_unique_count': 'count'})
    sloc_user_count.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
    return sloc_user_count


# 获取用户从某个地点出发的行为次数
@dump_feature
def get_global_user_sloc_count(start_day, end_day, label_day, prediction_type):
    train = train_all.copy()
    user_sloc_count = train.groupby(['userid','geohashed_start_loc'], as_index=False)['userid'].agg({'global_user_sloc_count': 'count'})
    user_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
    return user_sloc_count


# 获取用户从某个地点出发的行为比率
@dump_feature
def get_global_user_sloc_ratio(start_day, end_day, label_day, prediction_type):
    train = train_all.copy()
    user_sloc_count = train.groupby(['userid','geohashed_start_loc'])['userid'].agg({'global_user_sloc_count_ratio': 'count'})
    user_count = train.groupby(['userid'])['userid'].agg({'global_user_sloc_count_ratio': 'count'})  
    user_sloc_count = user_sloc_count.div(user_count).reset_index()
    user_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
    return user_sloc_count


# 获取目标地点的历史的小时热度(出发地)
@dump_feature
def get_global_sloc_hour_count(start_day, end_day, label_day, prediction_type):
    train = train_all.copy() 
    sloc_hour_count = train.groupby(['geohashed_start_loc', 'hour'], as_index=False)['userid'].agg({'global_sloc_hour_count': 'count'})
    sloc_hour_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'}, inplace=True)
    return sloc_hour_count

# 获取目标地点的历史的小时热度(出发地)
@dump_feature
def get_global_sloc_hour_ratio(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ] 
    sloc_hour_count = train.groupby(['geohashed_start_loc', 'hour'])['userid'].agg({'global_sloc_hour_count_ratio': 'count'})
    
    sloc_hour = train.groupby(['geohashed_start_loc'])['userid'].agg({'global_sloc_hour_count_ratio': 'count'})   
    sloc_hour_count = sloc_hour_count.div(sloc_hour).reset_index()
    
    sloc_hour_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'}, inplace=True)
    return sloc_hour_count



#################距离特征####################
# 获取用户的距离统计信息
@dump_feature
def get_user_distance_stat(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = calculate_single_distance(train, 'geohashed_start_loc', 'geohashed_end_loc')
    user_dis = train.groupby('userid',as_index=False)['distance'].agg({'user_dis_min':'min', 'user_dis_max':'max' ,'user_dis_mean':'mean'})
    return user_dis

# 获取用户的距离统计信息
@dump_feature
def get_user_distance_quantile_stat(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = calculate_single_distance(train, 'geohashed_start_loc', 'geohashed_end_loc') 
    user_dis1 = train.groupby('userid')['distance'].quantile(0.2).reset_index()
    user_dis1.columns = ['userid', 'user_dis_q2']
    user_dis2 = train.groupby('userid')['distance'].quantile(0.8).reset_index()
    user_dis2.columns = ['userid', 'user_dis_q8']
    user_disq = pd.merge(user_dis1, user_dis2, on='userid', how='left')
    return user_disq

@dump_feature
def get_bike_distance_stat(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = calculate_single_distance(train, 'geohashed_start_loc', 'geohashed_end_loc') 
    bike_dis = train.groupby('bikeid',as_index=False)['distance'].agg({'bike_dis_min':'min', 'bike_dis_max':'max' ,                                                                   'bike_dis_mean':'mean'})
    return bike_dis

# 获取出发地的距离统计信息
@dump_feature
def get_sloc_distance_stat(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = calculate_single_distance(train, 'geohashed_start_loc', 'geohashed_end_loc') 
    sloc_dis = train.groupby('geohashed_start_loc',as_index=False)['distance'].agg({'sloc_dis_min':'min', 'sloc_dis_max':'max' ,                                                                   'sloc_dis_mean':'mean'})
    sloc_dis.rename(columns={'geohashed_start_loc':'geohashed_end_loc'}, inplace=True)
    return sloc_dis

# 获取出发地的距离统计信息
@dump_feature
def get_real_sloc_distance_stat(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = calculate_single_distance(train, 'geohashed_start_loc', 'geohashed_end_loc') 
    sloc_dis = train.groupby('geohashed_start_loc',as_index=False)['distance'].agg({'sloc_real_dis_min':'min', 'sloc_real_dis_max':'max' , 'sloc_real_dis_mean':'mean'})
    return sloc_dis

# 获取目的地的距离统计信息
@dump_feature
def get_eloc_distance_stat(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = calculate_single_distance(train, 'geohashed_start_loc', 'geohashed_end_loc')
    eloc_dis = train.groupby('geohashed_end_loc',as_index=False)['distance'].agg({'eloc_dis_min':'min', 'eloc_dis_max':'max' ,                                                                   'eloc_dis_mean':'mean'})
    return eloc_dis

# 获取用户的距离统计信息
@dump_feature
def get_user_sloc_distance_stat(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = calculate_single_distance(train, 'geohashed_start_loc', 'geohashed_end_loc')
    user_sloc_dis = train.groupby(['userid', 'geohashed_start_loc'],as_index=False)['distance'].agg({'user_sloc_dis_min':'min', 'user_sloc_dis_max':'max' ,  'user_sloc_dis_mean':'mean'})
    user_sloc_dis.rename(columns={'geohashed_start_loc':'geohashed_end_loc'}, inplace=True)
    return user_sloc_dis

@dump_feature
def get_user_real_sloc_distance_stat(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = calculate_single_distance(train, 'geohashed_start_loc', 'geohashed_end_loc')
    user_sloc_dis = train.groupby(['userid', 'geohashed_start_loc'],as_index=False)['distance'].agg({'user_real_sloc_dis_min':'min', 'user_real_sloc_dis_max':'max' ,  'user_real_sloc_dis_mean':'mean'})
    return user_sloc_dis

# 获取用户的距离统计信息
@dump_feature
def get_user_eloc_distance_stat(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = calculate_single_distance(train, 'geohashed_start_loc', 'geohashed_end_loc') 
    user_eloc_dis = train.groupby(['userid', 'geohashed_end_loc'],as_index=False)['distance'].agg({'user_eloc_dis_min':'min', 'user_eloc_dis_max':'max' ,  'user_eloc_dis_mean':'mean'})
    return user_eloc_dis


################# 角度特征 ####################

# 获取用户的距离统计信息
@dump_feature
def get_user_sloc_angle_stat(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = calculate_single_angle(train, 'geohashed_start_loc', 'geohashed_end_loc')
    user_sloc_angle = train.groupby(['userid', 'geohashed_start_loc'],as_index=False)['bearing_array'].agg({'user_sloc_angle_min':'min', 'user_sloc_angle_max':'max' ,  'user_sloc_angle_mean':'mean'})
    return user_sloc_angle

# 获取用户的距离统计信息
@dump_feature
def get_user_eloc_angle_stat(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    train = calculate_single_angle(train, 'geohashed_start_loc', 'geohashed_end_loc')
    user_eloc_angle = train.groupby(['userid', 'geohashed_end_loc'],as_index=False)['bearing_array'].agg({'user_eloc_angle_min':'min', 'user_eloc_angle_max':'max' ,  'user_eloc_angle_mean':'mean'})
    return user_eloc_angle


################## 时间特征  ########################

# 获取目标地点的历史小时热度(目的地)
@dump_feature
def get_eloc_hour_mean(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]  
    eloc_hour_mean = train.groupby(['geohashed_end_loc'], as_index=False)['hour'].agg({'eloc_hour_mean': 'mean'})
    return eloc_hour_mean

# 获取目标地点的历史小时热度(目的地)
@dump_feature
def get_eloc_second_mean(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ] 
    train['second'] = train.starttime.dt.hour.values*3600 +  train.starttime.dt.minute.values*60+train.starttime.dt.second.values
    eloc_second_mean = train.groupby(['geohashed_end_loc'], as_index=False)['second'].agg({'eloc_second_mean': 'mean'})
    return eloc_second_mean


#################try  时间角度 地点刻画 矢量距离 #####################
# 获取用户从某个地点出发的行为比率
@dump_feature
def get_user_eloc_lasttime(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]
    train.sort_values(by='starttime', inplace=True)
    user_eloc_last = train.groupby(['userid','geohashed_end_loc'], as_index=False).last()[['userid','geohashed_end_loc', 'starttime']]
    user_eloc_last.rename(columns={'starttime': 'eloc_lasttime'}, inplace=True)
    return user_eloc_last

@dump_feature
def get_eloc_lasttime(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]
    train.sort_values(by='starttime', inplace=True)
    user_eloc_last = train.groupby(['geohashed_end_loc'], as_index=False).last()[['geohashed_end_loc', 'starttime']]
    user_eloc_last.rename(columns={'starttime': 'eloc_single_lasttime'}, inplace=True)
    return user_eloc_last

@dump_feature
def get_user_sloc_eloc_lasttime(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]
    train.sort_values(by='starttime', inplace=True)
    user_eloc_last = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'], as_index=False).last()[['userid', 'geohashed_start_loc', 'geohashed_end_loc', 'starttime']]
    user_eloc_last.rename(columns={'starttime': 'user_sloc_eloc_lasttime'}, inplace=True)
    return user_eloc_last

def get_user_sloc_lasttime(samples, start_day, end_day, label_day, prediction_type):
    train = train_all[(train_all.day<=end_day)][['userid', 'geohashed_start_loc', 'starttime']]
    train.columns = ['userid', 'geohashed_end_loc', 'starttime_']
    
    samples_ = samples[['orderid', 'starttime', 'userid', 'geohashed_end_loc']]
    samples_ = pd.merge(samples_, train, on=['userid', 'geohashed_end_loc'], how='left')
    samples_['sloc_lasttime'] = (pd.DatetimeIndex(samples_.starttime) - pd.DatetimeIndex(samples_.starttime_)).total_seconds().values
    samples_ = samples_[ (samples_.sloc_lasttime > 0) | (samples_.sloc_lasttime.isnull()) ]
    samples_.sort_values(by='sloc_lasttime', inplace=True)
    samples_ = samples_.groupby(['orderid'], as_index=False).first()[['orderid', 'sloc_lasttime']]
    samples = pd.merge(samples, samples_, on='orderid', how='left')
    
    return samples

def get_user_sloc_lasttime_(samples, start_day, end_day, label_day, prediction_type):
    train = train_all[(train_all.day<=end_day)][['userid', 'geohashed_start_loc', 'starttime']]
    train.columns = ['userid', 'geohashed_end_loc', 'starttime_']
    
    samples_ = samples[['orderid', 'starttime', 'userid', 'geohashed_end_loc']]
    print (samples_.shape, 'before')
    samples_ = pd.merge(samples_, train, on=['userid', 'geohashed_end_loc'], how='left')
    print (samples_.shape, 'after')
    samples_['sloc_lasttime'] = (pd.DatetimeIndex(samples_.starttime) - pd.DatetimeIndex(samples_.starttime_)).total_seconds().values
    samples_ = samples_[ samples_.sloc_lasttime > 0 ]
    
    samples_.sort_values(by='sloc_lasttime', inplace=True)
    samples_ = samples_.groupby(['orderid', 'geohashed_end_loc'], as_index=False).first()[['orderid','geohashed_end_loc', 'sloc_lasttime']]
    samples = pd.merge(samples, samples_, on=['orderid', 'geohashed_end_loc'], how='left')
    
    return samples

def get_user_sloc_today_times_before(samples, start_day, end_day, label_day, prediction_type):
    train = train_all[(train_all.day<=end_day)][['userid', 'geohashed_start_loc', 'starttime']]
    train.columns = ['userid', 'geohashed_end_loc', 'starttime_']
    
    samples_ = samples[['orderid', 'starttime', 'userid', 'geohashed_end_loc']]
    samples_ = pd.merge(samples_, train, on=['userid', 'geohashed_end_loc'], how='left')
    samples_['sloc_lasttime'] = (pd.DatetimeIndex(samples_.starttime) - pd.DatetimeIndex(samples_.starttime_)).total_seconds().values
    samples_ = samples_[ (samples_.sloc_lasttime > 0)]
    samples_ = samples_.groupby(['orderid', 'geohashed_end_loc'], as_index=False)['sloc_lasttime'].agg({'sloc_today_times_before_count':'count'})[['orderid', 'geohashed_end_loc', 'sloc_today_times_before_count']]
    samples = pd.merge(samples, samples_, on=['orderid', 'geohashed_end_loc'], how='left')
    return samples

def get_user_sloc_today_times_after(samples, start_day, end_day, label_day, prediction_type):
    train = train_all[(train_all.day>=end_day)][['userid', 'geohashed_start_loc', 'starttime']]
    train.columns = ['userid', 'geohashed_end_loc', 'starttime_']
    
    samples_ = samples[['orderid', 'starttime', 'userid', 'geohashed_end_loc']]
    samples_ = pd.merge(samples_, train, on=['userid', 'geohashed_end_loc'], how='left')
    samples_['sloc_lasttime'] = (pd.DatetimeIndex(samples_.starttime) - pd.DatetimeIndex(samples_.starttime_)).total_seconds().values
    samples_ = samples_[ (samples_.sloc_lasttime < 0)]
    samples_ = samples_.groupby(['orderid', 'geohashed_end_loc'], as_index=False)['sloc_lasttime'].agg({'sloc_today_times_after_count':'count'})[['orderid', 'geohashed_end_loc', 'sloc_today_times_after_count']]
    samples = pd.merge(samples, samples_, on=['orderid', 'geohashed_end_loc'], how='left')
    return samples

def get_user_sloc_nexttime(samples, start_day, end_day, label_day, prediction_type):
    train = train_all[(train_all.day==end_day)][['userid', 'geohashed_start_loc', 'starttime']]
    train.columns = ['userid', 'geohashed_end_loc', 'starttime_']
    
    samples_ = samples[['orderid', 'starttime', 'userid', 'geohashed_end_loc']]

    samples_ = pd.merge(samples_, train, on=['userid', 'geohashed_end_loc'], how='left')
    print (samples_.shape, 'after')
    samples_['sloc_nexttime'] = (pd.DatetimeIndex(samples_.starttime) - pd.DatetimeIndex(samples_.starttime_)).total_seconds().values
    samples_ = samples_[ samples_.sloc_nexttime < 0 ]
    samples_.sort_values(by='sloc_nexttime', inplace=True)
    samples_ = samples_.groupby(['orderid'], as_index=False).last()[['orderid', 'sloc_nexttime']]
    samples = pd.merge(samples, samples_, on='orderid', how='left')
    
    return samples

def get_user_sloc_nexttime_(samples, start_day, end_day, label_day, prediction_type):
    train = train_all[(train_all.day>=end_day)][['userid', 'geohashed_start_loc', 'starttime']]
    train.columns = ['userid', 'geohashed_end_loc', 'starttime_']
    
    samples_ = samples[['orderid', 'starttime', 'userid', 'geohashed_end_loc']]
    print (samples_.shape, 'before')
    samples_ = pd.merge(samples_, train, on=['userid', 'geohashed_end_loc'], how='left')
    print (samples_.shape, 'after')
    samples_['sloc_nexttime'] = (pd.DatetimeIndex(samples_.starttime) - pd.DatetimeIndex(samples_.starttime_)).total_seconds().values
    samples_ = samples_[ (samples_.sloc_nexttime < 0) ]
    samples_.sort_values(by='sloc_nexttime', inplace=True)
    samples_ = samples_.groupby(['orderid', 'geohashed_end_loc'], as_index=False).last()[['orderid', 'geohashed_end_loc','sloc_nexttime']]
    samples = pd.merge(samples, samples_, on=['orderid', 'geohashed_end_loc'], how='left')
    
    return samples



# 获取用户从某个地点出发的行为比率
@dump_feature
def get_user_samesloc_eloc(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]
    train.sort_values(by='starttime', inplace=True)
    user_samesloc_lasteloc = train.groupby(['userid','geohashed_start_loc'], as_index=False).last()[['userid','geohashed_start_loc', 'geohashed_end_loc']]
    user_samesloc_lasteloc.rename(columns={'geohashed_end_loc': 'samesloc_lasteloc'}, inplace=True)
    return user_samesloc_lasteloc

@dump_feature
def get_user_samesloc_frequentest_eloc(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ]
    train = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc' ], as_index=False)['userid'].agg({'user_sloc_eloc_count':'count'})
    train.sort_values(by='user_sloc_eloc_count', inplace=True)
    user_samesloc_frequentest_eloc = train.groupby(['userid','geohashed_start_loc'], as_index=False).last()[['userid','geohashed_start_loc', 'geohashed_end_loc']]
    user_samesloc_frequentest_eloc.rename(columns={'geohashed_end_loc': 'samesloc_frequentest_eloc'}, inplace=True)
    return user_samesloc_frequentest_eloc

@dump_feature
def get_user_most_frequent_eloc(start_day, end_day, label_day, prediction_type):
    user_eloc_count = get_user_eloc_count(start_day, end_day, label_day, prediction_type)
    user_eloc_count.sort_values(by='user_eloc_count', inplace=True, ascending=True)
    user_eloc_count =  user_eloc_count.groupby(['userid'], as_index=False).last()[['userid', 'geohashed_end_loc']]
    user_eloc_count.columns = ['userid', 'user_most_freq_eloc']
    return user_eloc_count

@dump_feature
def get_sloc_most_frequent_eloc(start_day, end_day, label_day, prediction_type):
    sloc_eloc_count = get_sloc_eloc_count(start_day, end_day, label_day, prediction_type)
    sloc_eloc_count.sort_values(by='sloc_eloc_count', inplace=True, ascending=True)
    sloc_eloc_count =  sloc_eloc_count.groupby(['geohashed_start_loc'], as_index=False).last()[['geohashed_start_loc', 'geohashed_end_loc']]
    sloc_eloc_count.columns = ['geohashed_start_loc', 'sloc_most_freq_eloc']
    return sloc_eloc_count

@dump_feature
def get_user_eloc_center(start_day, end_day, label_day, prediction_type):
    train = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ][['userid', 'geohashed_end_loc']]
    loc_lat_lon =  get_loc_lat_lon()
    loc_lat_lon.columns = ['geohashed_end_loc', 'elat', 'elon']
    train = pd.merge(train, loc_lat_lon, on='geohashed_end_loc', how='left')[['userid', 'elat', 'elon']]
    user_eloc_center = train.groupby('userid', as_index=False).mean()
    user_eloc_center.columns = ['userid', 'user_eloc_lat_center', 'user_eloc_lon_center']
    return user_eloc_center

@dump_feature
def get_user_sloc_center(start_day, end_day, label_day, prediction_type):
    train = train_all[['userid', 'geohashed_start_loc']]
    loc_lat_lon =  get_loc_lat_lon()
    loc_lat_lon.columns = ['geohashed_start_loc', 'slat', 'slon']
    train = pd.merge(train, loc_lat_lon, on='geohashed_start_loc', how='left')[['userid', 'slat', 'slon']]
    user_sloc_center = train.groupby('userid', as_index=False).mean()
    user_sloc_center.columns = ['userid', 'user_sloc_lat_center', 'user_sloc_lon_center']
    return user_sloc_center

@dump_feature
def get_user_loc_center(start_day, end_day, label_day, prediction_type):
    train_eloc = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ][['userid', 'geohashed_end_loc']]
    train_sloc = train_all[ (train_all.day>=start_day) & (train_all.day<=end_day) ][['userid', 'geohashed_start_loc']]
    train_sloc.columns = ['userid', 'geohashed_end_loc']
    
    train = pd.concat([train_eloc, train_sloc], axis=0)
    loc_lat_lon_ = loc_lat_lon.copy()
    loc_lat_lon_.columns = ['geohashed_end_loc', 'lat', 'lon']
    train = pd.merge(train, loc_lat_lon_, on='geohashed_end_loc', how='left')[['userid',  'lat', 'lon']]
    user_loc_center = train.groupby('userid', as_index=False).mean()
    user_loc_center.columns = ['userid', 'user_loc_lat_center', 'user_loc_lon_center']
    
    return user_loc_center
#################try#####################


########## 计算两点之间的距离、方位角之类的变量 ###############

# 计算两点之间距离
def haversine(lat1, lng1, lat2, lng2):
    """function to calculate haversine distance between two co-ordinates"""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return(h)

def manhattan(lat1, lng1, lat2, lng2):
    """function to calculate manhatten distance between pick_drop"""
    a = haversine(lat1, lng1, lat1, lng2)
    b = haversine(lat1, lng1, lat2, lng1)
    return a + b

import math
def bearing_array(lat1, lng1, lat2, lng2):
    """ function was taken from beluga's notebook as this function works on array
    while my function used to work on individual elements and was noticably slow"""
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def string_sim(loc1, loc2):
    sims = 0
    for i in range(len(loc1)):
        if loc1[i] == loc2[i]:
            sims += 1
        else:
            break
    return sims
    
@dump_feature
def get_loc_lat_lon():
    locs = list(set(train_all['geohashed_start_loc']) | set(train_all['geohashed_end_loc']))
    if np.nan in locs:
        locs.remove(np.nan)
    deloc = []
    for loc in locs:
        deloc.append(Geohash.decode_exactly(loc))
    loc_dict = dict(zip(locs,deloc))
        
    lat_lon = pd.DataFrame([[key, values[0], values[1]] for (key, values) in loc_dict.iteritems()],columns=['loc', 'lat', 'lon'])
    return lat_lon 

def calculate_points_relations(samples, feat1, feat2, prefix, use_type=False, id_kind1='', id_kind2=''):
    print ('calculate_points_relations')
    t_start = time.time()
    
    lat_lon = get_loc_lat_lon()
    lat_lon.columns = [feat1, 'sloc_lat', 'sloc_lon']
    samples = pd.merge(samples, lat_lon, on=feat1, how='left')
        
    lat_lon.columns = [feat2,  'eloc_lat', 'eloc_lon']
    samples = pd.merge(samples, lat_lon, on=feat2, how='left')
        
    samples[prefix+'distance'] = haversine(samples.sloc_lat.values, samples.sloc_lon.values, samples.eloc_lat.values, samples.eloc_lon.values)
    
    samples[prefix+'lat_diff'] = samples.sloc_lat.values -  samples.eloc_lat.values
    samples[prefix+'lon_diff'] = samples.sloc_lon.values -  samples.eloc_lon.values
    samples[prefix+'manhattan_distance'] = manhattan(samples.sloc_lat.values, samples.sloc_lon.values, samples.eloc_lat.values, samples.eloc_lon.values)
    samples[prefix+'bearing_array'] = bearing_array(samples.sloc_lat.values, samples.sloc_lon.values, samples.eloc_lat.values, samples.eloc_lon.values)    
    
    if use_type == True:
        samples[prefix+'speed'] = samples[prefix+'distance'].values / samples[id_kind2+'_order_time_diff'].values
        samples[prefix+'lat_speed'] = samples[prefix+'lat_diff'].values / samples[id_kind2+'_order_time_diff'].values 
        samples[prefix+'lon_speed'] = samples[prefix+'lon_diff'].values / samples[id_kind2+'_order_time_diff'].values
        samples[prefix+'manhattan_speed'] = samples[prefix+'manhattan_distance'].values / samples[id_kind2+'_order_time_diff'].values
        
        if 'user' in id_kind1:
            samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'distance'] = -1000000
            samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'lat_diff'] = -1000000
            samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'lon_diff'] = -1000000
            samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'manhattan_distance'] = -1000000
            samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'bearing_array'] = -1000000

            samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'speed'] = -1000000
            samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'lat_speed'] = -1000000
            samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'lon_speed'] = -1000000
            samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'manhattan_speed'] = -1000000
            
        if 'bike' in id_kind1:
            samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'distance'] = -1000000
            samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'lat_diff'] = -1000000
            samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'lon_diff'] = -1000000
            samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'manhattan_distance'] = -1000000
            samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'bearing_array'] = -1000000

            samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'speed'] = -1000000
            samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'lat_speed'] = -1000000
            samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'lon_speed'] = -1000000
            samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'manhattan_speed'] = -1000000

        if 'next' in id_kind1:
            samples[prefix+'speed_'] = samples['distance'].values / samples[id_kind2+'_order_time_diff'].values
            samples[prefix+'lat_speed_'] = samples['lat_diff'].values / samples[id_kind2+'_order_time_diff'].values 
            samples[prefix+'lon_speed_'] = samples['lon_diff'].values / samples[id_kind2+'_order_time_diff'].values
            samples[prefix+'manhattan_speed_'] = samples['manhattan_distance'].values / samples[id_kind2+'_order_time_diff'].values
            
            if 'user' in id_kind1:
                samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'speed_'] = -1000000
                samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'lat_speed_'] = -1000000
                samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'lon_speed_'] = -1000000
                samples.loc[samples.userid!=samples[id_kind1+'id'], prefix+'manhattan_speed_'] = -1000000
            if 'bike' in id_kind1:
                samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'speed_'] = -1000000
                samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'lat_speed_'] = -1000000
                samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'lon_speed_'] = -1000000
                samples.loc[samples.bikeid!=samples[id_kind1+'id'], prefix+'manhattan_speed_'] = -1000000
            
        if 'user' in id_kind1:
            samples.loc[samples.userid!=samples[id_kind1+'id'], id_kind2+'_order_time_diff'] = -1000000
        if 'bike' in id_kind1:
            samples.loc[samples.bikeid!=samples[id_kind1+'id'], id_kind2+'_order_time_diff'] = -1000000
    
    samples.drop(['sloc_lat', 'sloc_lon', 'eloc_lat', 'eloc_lon'] ,axis=1, inplace=True)
    gc.collect()
    t_end = time.time() 
    print ('call %s() in %fs' % ('calculate_points_relations', (t_end - t_start)))
    return samples


def calculate_single_distance(samples, feat1, feat2):
    print 
    print ('calculate_single_distance')
    print (datetime.now())
    
    lat_lon = get_loc_lat_lon()
    lat_lon.columns = [feat1, 'sloc_lat', 'sloc_lon']
    samples = pd.merge(samples, lat_lon, on=feat1, how='left')
        
    lat_lon.columns = [feat2,  'eloc_lat', 'eloc_lon']
    samples = pd.merge(samples, lat_lon, on=feat2, how='left')
        
    samples['distance'] = haversine(samples.sloc_lat.values, samples.sloc_lon.values, samples.eloc_lat.values,  samples.eloc_lon.values) 
        
    samples.drop(['sloc_lat', 'sloc_lon', 'eloc_lat', 'eloc_lon'] ,axis=1, inplace=True)
    gc.collect()
    return samples

def calculate_single_angle(samples, feat1, feat2):
    print 
    print ('calculate_single_distance')
    print (datetime.now())
    
    lat_lon = get_loc_lat_lon()
    lat_lon.columns = [feat1, 'sloc_lat', 'sloc_lon']
    samples = pd.merge(samples, lat_lon, on=feat1, how='left')
        
    lat_lon.columns = [feat2,  'eloc_lat', 'eloc_lon']
    samples = pd.merge(samples, lat_lon, on=feat2, how='left')
        
    samples['bearing_array'] = bearing_array(samples.sloc_lat.values, samples.sloc_lon.values, samples.eloc_lat.values,  samples.eloc_lon.values) 
        
    samples.drop(['sloc_lat', 'sloc_lon', 'eloc_lat', 'eloc_lon'] ,axis=1, inplace=True)
    gc.collect()
    return samples


    
############## 制作训练集和测试集 ###############
def make_set(end_day, prediction_type):
    dump_path = cache_path + 'dataset_samples_%s_%s_27_fan.hdf' %(end_day, prediction_type)
    print ()
    if os.path.exists(dump_path):
        samples = pd.read_hdf(dump_path, 'w')          
        return samples 
    
    else:
        print (end_day, prediction_type)
        print('开始构造样本...')
        samples = get_sample(end_day, prediction_type)
        samples = calculate_points_relations(samples, 'geohashed_start_loc', 'geohashed_end_loc', '')
        samples = samples[samples.distance<8]
        gc.collect()
        print('开始构造特征...')
        
        ############leak特征################
        # 用户leak
        samples = pd.merge(samples, get_orderid_user_leak(), on='orderid', how='left')
        samples = calculate_points_relations(samples, 'geohashed_end_loc', 'user_last_sloc', 'eloc_user_last_sloc_', True, 'last_user', 'user_last')
        samples = calculate_points_relations(samples, 'geohashed_end_loc', 'user_next_sloc', 'eloc_user_next_sloc_', True, 'next_user', 'user_next')
                   
        samples.drop(['last_userid',  'next_userid', 'user_last_sloc',  'user_next_sloc'], axis=1, inplace=True)
        gc.collect()    
        
        # 自行车leak
        samples = pd.merge(samples, get_orderid_bike_leak(), on='orderid', how='left')
        samples = calculate_points_relations(samples, 'geohashed_end_loc', 'bike_next_sloc', 'eloc_bike_next_sloc_', True, 'next_bike', 'bike_next')
        
        samples['bike_next_sloc_same'] = samples['bike_next_sloc'] == samples['geohashed_end_loc']
        samples.drop(['next_bikeid', 'bike_next_sloc'], axis=1, inplace=True)
        gc.collect()
        
        ##########未交叉特征######################
        samples = pd.merge(samples, get_user_count(10, end_day, end_day, prediction_type), on=['userid'], how='left')
        samples = pd.merge(samples, get_bike_count(10, end_day, end_day, prediction_type), on=['bikeid'], how='left')
        samples = pd.merge(samples, get_sloc_count(10, end_day, end_day, prediction_type), on=['geohashed_end_loc'], how='left')  
        samples = pd.merge(samples, get_eloc_count(10, end_day-prediction_type,end_day, prediction_type), on=['geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_bike_unqiue_user_count(10, end_day, end_day, prediction_type), on=['bikeid'], how='left')
        samples = pd.merge(samples, get_eloc_unique_user_count(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_sloc_unique_user_count(10, end_day, end_day, prediction_type), on=['geohashed_end_loc'], how='left')
        gc.collect()
        
        ############二交叉特征####################
        samples = pd.merge(samples, get_user_sloc_count(10, end_day, end_day, prediction_type), on=['userid', 'geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_user_sloc_ratio(10, end_day, end_day, prediction_type), on=['userid', 'geohashed_end_loc'], how='left')  
        
        samples = pd.merge(samples, get_user_eloc_count(10, end_day-prediction_type, end_day, prediction_type), on=['userid', 'geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_user_eloc_ratio(10, end_day-prediction_type, end_day, prediction_type), on=['userid', 'geohashed_end_loc'], how='left')        
        
        samples = pd.merge(samples, get_sloc_eloc_count(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_start_loc', 'geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_sloc_eloc_ratio(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_start_loc', 'geohashed_end_loc'], how='left')    
        
        samples = pd.merge(samples, get_eloc_sloc_count(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_start_loc', 'geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_eloc_sloc_ratio(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_start_loc', 'geohashed_end_loc'], how='left')        
        
        samples = pd.merge(samples, get_sloc_today_hour_count(), on=['geohashed_end_loc', 'date', 'hour'], how='left')  
        samples = pd.merge(samples, get_sloc_hour_count(10, end_day, end_day, prediction_type), on=['geohashed_end_loc', 'hour'], how='left')  
        samples = pd.merge(samples, get_sloc_hour_ratio(10, end_day, end_day, prediction_type), on=['geohashed_end_loc', 'hour'], how='left')         
        samples = pd.merge(samples, get_eloc_hour_ratio(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_end_loc', 'hour'], how='left')
        gc.collect()
        
        ############三交叉特征#####################
        samples = pd.merge(samples, get_user_sloc_eloc_count(10, end_day-prediction_type, end_day, prediction_type), on=['userid', 'geohashed_start_loc', 'geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_user_eloc_sloc_count(10, end_day-prediction_type, end_day, prediction_type), on=['userid', 'geohashed_start_loc', 'geohashed_end_loc'], how='left')        
        samples = pd.merge(samples, get_user_hour_eloc_count(10, end_day-prediction_type, end_day, prediction_type), on=['userid', 'hour', 'geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_user_hour_sloc_count(10, end_day, end_day, prediction_type), on=['userid', 'hour', 'geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_sloc_hour_eloc_count(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_start_loc', 'hour', 'geohashed_end_loc'], how='left')              
        samples = pd.merge(samples, get_eloc_hour_sloc_count(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_start_loc', 'hour', 'geohashed_end_loc'], how='left')  
        gc.collect()
        
        
        ############四交叉特征######################
        samples = pd.merge(samples,get_user_hour_sloc_eloc_count(10, end_day-prediction_type, end_day, prediction_type), on=['userid',  'hour', 'geohashed_start_loc', 'geohashed_end_loc'], how='left')
        
        
        ################距离特征####################
        samples = pd.merge(samples, get_user_distance_stat(10, end_day-prediction_type, end_day, prediction_type), on=['userid'], how='left')        
        samples.loc[:, 'user_dis_min'] = samples['user_dis_min'] - samples['distance']
        samples.loc[:, 'user_dis_max'] = samples['user_dis_max'] - samples['distance']
        samples.loc[:, 'user_dis_mean'] = samples['user_dis_mean'] - samples['distance']
        
        samples = pd.merge(samples, get_bike_distance_stat(10, end_day-prediction_type, end_day, prediction_type), on=['bikeid'], how='left')        
        samples.loc[:, 'bike_dis_min'] = samples['bike_dis_min'] - samples['distance']
        samples.loc[:, 'bike_dis_max'] = samples['bike_dis_max'] - samples['distance']
        samples.loc[:, 'bike_dis_mean'] = samples['bike_dis_mean'] - samples['distance']        
        
        samples = pd.merge(samples, get_sloc_distance_stat(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_end_loc'], how='left')    
        samples.loc[:, 'sloc_dis_min'] = samples['sloc_dis_min'] - samples['distance']
        samples.loc[:, 'sloc_dis_max'] = samples['sloc_dis_max'] - samples['distance']
        samples.loc[:, 'sloc_dis_mean'] = samples['sloc_dis_mean'] - samples['distance']
        
        samples = pd.merge(samples, get_eloc_distance_stat(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_end_loc'], how='left')        
        samples.loc[:, 'eloc_dis_min'] = samples['eloc_dis_min'] - samples['distance']
        samples.loc[:, 'eloc_dis_max'] = samples['eloc_dis_max'] - samples['distance']
        samples.loc[:, 'eloc_dis_mean'] = samples['eloc_dis_mean'] - samples['distance']  
        
        samples = pd.merge(samples, get_user_sloc_distance_stat(10, end_day-prediction_type, end_day, prediction_type), on=['userid','geohashed_end_loc'], how='left')        
        samples.loc[:, 'user_sloc_dis_min'] = samples['user_sloc_dis_min'] - samples['distance']
        samples.loc[:, 'user_sloc_dis_max'] = samples['user_sloc_dis_max'] - samples['distance']
        samples.loc[:, 'user_sloc_dis_mean'] = samples['user_sloc_dis_mean'] - samples['distance']   
        
        samples = pd.merge(samples, get_user_eloc_distance_stat(10, end_day-prediction_type, end_day, prediction_type), on=['userid','geohashed_end_loc'], how='left')        
        samples.loc[:, 'user_eloc_dis_min'] = samples['user_eloc_dis_min'] - samples['distance']
        samples.loc[:, 'user_eloc_dis_max'] = samples['user_eloc_dis_max'] - samples['distance']
        samples.loc[:, 'user_eloc_dis_mean'] = samples['user_eloc_dis_mean'] - samples['distance']  
        gc.collect()
       
        ################### 时间特征、全局统计特诊、 #########################

        samples = pd.merge(samples, get_user_eloc_lasttime(10, end_day-prediction_type, end_day, prediction_type), on=['userid', 'geohashed_end_loc'], how='left')
        samples['eloc_lasttime'] = (pd.DatetimeIndex( samples.starttime) - pd.DatetimeIndex( samples.eloc_lasttime)).total_seconds().values

        samples = pd.merge(samples, get_user_distance_quantile_stat(10, end_day-prediction_type, end_day, prediction_type), on=['userid'], how='left')        
        samples.loc[:, 'user_dis_q2'] = samples['user_dis_q2'] - samples['distance']
        samples.loc[:, 'user_dis_q8'] = samples['user_dis_q8'] - samples['distance']
        gc.collect()
    
        samples = pd.merge(samples, get_global_sloc_count(10, end_day, end_day, prediction_type), on=['geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_global_sloc_unique_user_count(10, end_day, end_day, prediction_type), on=[ 'geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_global_user_sloc_count(10, end_day, end_day, prediction_type), on=['userid', 'geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_global_user_sloc_ratio(10, end_day, end_day, prediction_type), on=['userid', 'geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_global_sloc_hour_count(10, end_day, end_day, prediction_type), on=['geohashed_end_loc', 'hour'], how='left')
        samples = pd.merge(samples, get_global_sloc_hour_ratio(10, end_day, end_day, prediction_type), on=['geohashed_end_loc', 'hour'], how='left')   
        gc.collect()
        
        ################### 地点刻画 #########################        
        samples = pd.merge(samples, get_user_samesloc_eloc(10, end_day-prediction_type, end_day, prediction_type), on=['userid', 'geohashed_start_loc'], how='left')
        samples = calculate_points_relations(samples, 'geohashed_end_loc', 'samesloc_lasteloc', 'samesloc_lasteloc_')
        samples.drop('samesloc_lasteloc', axis=1, inplace=True)

        samples = pd.merge(samples, get_user_samesloc_frequentest_eloc(10, end_day-prediction_type, end_day, prediction_type), on=['userid', 'geohashed_start_loc'], how='left')
        samples = calculate_points_relations(samples,'geohashed_end_loc', 'samesloc_frequentest_eloc', 'samesloc_frequentest_eloc_')   
        samples.drop('samesloc_frequentest_eloc', axis=1, inplace=True)

        samples = pd.merge(samples, get_user_most_frequent_eloc(10, end_day-prediction_type, end_day, prediction_type), on='userid', how='left')  
        samples = calculate_points_relations(samples,'geohashed_end_loc', 'user_most_freq_eloc', 'user_most_freq_eloc_')
        samples.drop('user_most_freq_eloc', axis=1, inplace=True)

        samples = pd.merge(samples, get_sloc_most_frequent_eloc(10, end_day-prediction_type, end_day, prediction_type), on='geohashed_start_loc', how='left')   
        samples = calculate_points_relations(samples,'geohashed_end_loc', 'sloc_most_freq_eloc', 'sloc_most_freq_eloc_')
        samples.drop('sloc_most_freq_eloc', axis=1, inplace=True)
        gc.collect()
        
        ################### 查漏补缺 #########################  
        samples = pd.merge(samples, get_eloc_hour_count(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_end_loc', 'hour'], how='left')
        samples = get_user_sloc_nexttime(samples, 10, end_day, end_day, prediction_type)
        
        
        samples = pd.merge(samples, get_user_real_sloc_distance_stat(10, end_day-prediction_type, end_day, prediction_type), on=['userid','geohashed_start_loc'], how='left')    
        samples.loc[:, 'user_real_sloc_dis_min'] = samples['user_real_sloc_dis_min'] - samples['distance']
        samples.loc[:, 'user_real_sloc_dis_max'] = samples['user_real_sloc_dis_max'] - samples['distance']
        samples.loc[:, 'user_real_sloc_dis_mean'] = samples['user_real_sloc_dis_mean'] - samples['distance']
        
        samples = pd.merge(samples, get_eloc_hour_mean(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_end_loc'], how='left')  
        samples = pd.merge(samples, get_eloc_second_mean(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_end_loc'], how='left')  
        
        samples = pd.merge(samples, get_cluster_sloc_eloc_count(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_start_loc', 'geohashed_end_loc'], how='left')
        samples = pd.merge(samples, get_cluster_sloc_eloc_ratio(10, end_day-prediction_type, end_day, prediction_type), on=['geohashed_start_loc', 'geohashed_end_loc'], how='left')         
        samples = pd.merge(samples, get_user_cluster_sloc_eloc_count(10, end_day-prediction_type, end_day, prediction_type), on=['userid', 'geohashed_start_loc', 'geohashed_end_loc'], how='left')       
        samples = pd.merge(samples, get_user_sloc_eloc_count_ratio(10, end_day-prediction_type, end_day, prediction_type), on=['userid', 'geohashed_start_loc', 'geohashed_end_loc'], how='left')   
        samples = pd.merge(samples, get_user_cluster_sloc_eloc_count_ratio(10, end_day-prediction_type, end_day, prediction_type), on=['userid', 'geohashed_start_loc', 'geohashed_end_loc'], how='left') 
        
        print (end_day, samples.shape, 'filter before')
        samples = samples.loc[samples.distance <= 7]
        print (end_day, samples.shape, 'filter after')
        
        gc.collect()
        
        samples = get_label(samples)
        
        print (end_day, samples.shape, 'prepare to dump')
        
        samples.to_hdf(dump_path, 'w')
        print (end_day, samples.shape, 'dump done')
        del samples; gc.collect()
        
        return pd.DataFrame([1])
    
def extract_feature_single():
    print ('start extract feature')
    funcs_list = []
    
    train_dates = list(range(25-use_days_to_train, 25))
    train_prediction_types =  [1] * len(train_dates)
    
    if predict == True:
        test_prediction_types = list(range(1, 1 + days_to_predict))
        test_dates =  list(range(25, 25 + days_to_predict))
        date_prediction_type_pairs = list(zip(train_dates, train_prediction_types)) + zip(test_dates, test_prediction_types)
    else:
        date_prediction_type_pairs = list(zip(train_dates, train_prediction_types))
    
    for date, prediction_type in date_prediction_type_pairs:
        print ( date, prediction_type )
        funcs_list.append( [make_set, (date, prediction_type)] )
       
    applyParallel_feature(iter(funcs_list), execu, sample_thread ) 
    print ('data extract feature finish.')   
    
    
    
################### lgb ######################
def evaluate_predict_single(prediction_type):
    
    results = []
    no_predictors = ['orderid', 'geohashed_end_loc', 'userid', 'bikeid', 'starttime', u'geohashed_start_loc', u'label', u'date', u'day', 'dayofweek']
    
    if 1:
        train_dates = list(range(25-use_days_to_train, 25))
        predict_types = [prediction_type] * len(train_dates)
        date_prediction_type_pairs = zip(train_dates, predict_types)
        
        print (date_prediction_type_pairs)
        
        train_feat =  pd.concat(map(lambda x: make_set(*x), date_prediction_type_pairs), axis=0)
        gc.collect()
        training_label = train_feat.label.values
        train_feat.drop(no_predictors, axis=1, inplace=True)
        print (train_feat.columns)
        gc.collect()

        X_train, X_test, y_train, y_test = train_test_split(train_feat.astype(float), training_label, test_size= 0.1, random_state=0)
        del train_feat; gc.collect()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        del X_train, y_train; gc.collect()
        dval  = xgb.DMatrix(X_test,  label=y_test)
        del X_test, y_test; gc.collect()


        params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_estimators' : 1000,
                'learning_rate' : 0.12,
                'eta': 0.08,
                'max_depth': 7,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'scale_pos_weight': 1,
                'min_child_weight': 5,
                'verbose_eval': 1,
                'nthread': 32,
                'seed': 201709,
                'silent': 1,
        }

        evallist = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(params, dtrain, 2000, evallist, early_stopping_rounds=13)

        del dtrain, dval; gc.collect()
        model.save_model('xgb.model')
    
            
    if predict == True:
        
        test_dates = list(range(25, 25+days_to_predict))
        predict_types = range(1, 1+days_to_predict)
        date_prediction_type_pairs = zip(test_dates, predict_types)
        
        for day, pred_type in date_prediction_type_pairs:
            
            print (datetime.now())
            test_feat = make_set(day, pred_type)
            sub_training_data = test_feat.drop(no_predictors, axis=1).astype(float)
            test_feat_ = test_feat[['orderid', 'geohashed_end_loc']]
            del test_feat; gc.collect()
            dtest = xgb.DMatrix(sub_training_data)
            del sub_training_data; gc.collect()
            test_feat_.loc[:,'pred'] = model.predict(dtest)
            del dtest; gc.collect()
            
            results.append(test_feat_)

            print ('predict', day, pred_type, 'done' )
            print (datetime.now())

        return pd.concat(results, axis=0)
    else:
        return pd.DataFrame([1])


if pre_extract == 0:
    data =  get_orderid_user_leak()
    del data; gc.collect()

    data =  get_orderid_bike_leak()
    del data; gc.collect()
    
    data = get_sloc_today_hour_count()
    del data; gc.collect()

    data = get_label([])
    del data; gc.collect()
    
    del train_all_2week; gc.collect()
    
if count == 0:
    print (datetime.now())
    print ('start extract_feature()') 
    extract_feature_single()
    print ('extract_feature() done, 用时%s秒' %(time.time()-t0))
    
if eval_ == True:
    print (datetime.now())
    print ('start evaluate_predict ')
    results = evaluate_predict_single(1)
    print (datetime.now())

if predict == True:
    print ('finish evaluate_predict ')
    print ("the final result prob's shape: (%s, %s)" %(results.shape[0], results.shape[1]) )
    results.to_pickle('xgb.pickle')
print('一共用时{}秒'.format(time.time()-t0))
print (datetime.now())