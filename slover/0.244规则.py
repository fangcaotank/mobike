#coding:utf-8
import pandas as pd
import numpy as np
import Geohash as geo

# geohash =》 经纬度
def get_x(pos):
    x,y,x_m,y_m = geo.decode_exactly(str(pos))
    return x

def get_y(pos):
    x,y,x_m,y_m = geo.decode_exactly(str(pos))
    return y

# 根据经纬度计算距离
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

########################################################
# # 时间段的划分  22-5 6-13 14-21
def hour_map_all(hour):
    return 1

def hour_map_s(hour):
    # return hour
    hour = int(hour)
    if (hour >= 0) & (hour <= 6):
        return 0
    elif (hour >= 23) & (hour <= 23):
        return 0
    elif (hour >= 7) & (hour <= 9):
        return 1
    elif (hour >= 10) & (hour <= 16):
        return 3
    elif (hour >= 17) & (hour <= 22):
        return 2

# 时间段的划分
def hour_map(hour):
    hour = int(hour)
    if (hour >= 1) & (hour <= 6):
        return 1
    elif (hour == 0):
        return 3
    elif (hour >= 7) & (hour <= 12):
        return 1
    elif (hour >= 13) & (hour <= 16):
        return 2
    elif (hour >= 17) & (hour <= 23):
        return 3

def calc_km(df,km):
    print u'calc_km'
    df = df.fillna('wx4e5ye')
    df['start_x'] = df['geohashed_start_loc'].map(get_x)
    df['start_y'] = df['geohashed_start_loc'].map(get_y)

    df['end_x_1'] = df['end0'].map(get_x)
    df['end_y_1'] = df['end0'].map(get_y)

    df['end_x_2'] = df['end1'].map(get_x)
    df['end_y_2'] = df['end1'].map(get_y)

    df['end_x_3'] = df['end2'].map(get_x)
    df['end_y_3'] = df['end2'].map(get_y)

    df['hafuman_km_1'] = haversine_np(df['start_x'], df['start_y'], df['end_x_1'], df['end_y_1'])
    df['hafuman_km_2'] = haversine_np(df['start_x'], df['start_y'], df['end_x_2'], df['end_y_2'])
    df['hafuman_km_3'] = haversine_np(df['start_x'], df['start_y'], df['end_x_3'], df['end_y_3'])

    ss_1 = df[['orderid', 'geohashed_start_loc', 'end0', 'end1', 'end2']].set_index(
        ['orderid', 'geohashed_start_loc']).stack().reset_index()
    ss_2 = df[['orderid', 'geohashed_start_loc', 'hafuman_km_1', 'hafuman_km_2', 'hafuman_km_3']].set_index(
        ['orderid', 'geohashed_start_loc']).stack().reset_index()
    ss_l = ss_1['level_2']
    del ss_1['level_2']
    del ss_2['level_2']
    ss_1.rename(inplace=True, columns={0: 'loc'})
    ss_2.rename(inplace=True, columns={0: 'km'})
    ss = pd.concat([ss_1, ss_2['km']], axis=1)
    ss = ss.sort_values(['orderid', 'km'])
    print ss
    ss.loc[ss['km'] >= km, 'loc'] = np.nan
    ss = pd.concat([ss, ss_l], axis=1)
    ss = ss[['orderid', 'geohashed_start_loc', 'loc', 'level_2']].set_index(
        ['orderid', 'geohashed_start_loc', 'level_2']).unstack(level=-1).reset_index()
    test = pd.read_csv('../data/test.csv')[['orderid']]
    sub = pd.merge(test, ss, on=['orderid'], how='left')
    print sub[['orderid','geohashed_start_loc','end0','end1','end2']].head()
    return sub[['orderid','geohashed_start_loc','end0','end1','end2']]

##########################################################################################
# 1 用户-起点 集合
def get_userid_start_5_end(n,length,df_train):
    print u'1 == base + start(%d) + userid + top %d' %(length, n)
    df_train['geohashed_start_loc'] = df_train['geohashed_start_loc'].map(lambda x: x[:length])
    user_loc = df_train[['userid', 'geohashed_start_loc', 'geohashed_end_loc', 'orderid']].groupby(
        ['userid', 'geohashed_start_loc',  'geohashed_end_loc'], as_index=False).count()
    user_loc_rank = user_loc.sort_values(['userid', 'orderid'], ascending=False)
    return user_loc_rank.groupby(['userid', 'geohashed_start_loc'], as_index=False).nth(n)[
        ['userid',  'geohashed_end_loc', 'geohashed_start_loc']]

# 根据时间段合并用户对应的地点
def merge_get_userid_start_5_end(number,length,df_test,df_train,offset):
    df_test['start_true'] = df_test['geohashed_start_loc']
    df_test['geohashed_start_loc'] = df_test['geohashed_start_loc'].map(lambda x: x[:length])
    for n in range(number):
        df_test = pd.merge(df_test, get_userid_start_5_end(n+offset,length,df_train), on=['userid', 'geohashed_start_loc'],
                           how='left')
        df_test.rename(inplace=True, columns={'geohashed_end_loc': 'end%s' % (n)})
    del df_test['geohashed_start_loc']
    df_test.rename(inplace=True, columns={'start_true': 'geohashed_start_loc'})
    print u'merge-finish'
    return df_test

##########################################################################################
# 2 用户+ 起点 集合
def cov_1(hour):
    hour = int(hour)
    if (hour >= 7) & (hour <= 11):
        return 1
    elif (hour >= 16) & (hour <= 21):
        return 4
    else:
        return 0

def cov_2(hour):
    hour = int(hour)
    if (hour >= 7) & (hour <= 11):
        return 4
    elif (hour >= 16) & (hour <= 21):
        return 1
    else:
        return 0

def get_userid_start(n,length,df_train):
    print u'3 == user_start_cov %d' %(n)
    df_train['geohashed_start_loc'] = df_train['geohashed_start_loc'].map(lambda x: x[:length])
    df_train['hour'] = pd.to_datetime(df_train['starttime']).map(lambda x: x.strftime('%H'))
    df_train['hour'] = df_train['hour'].map(cov_1)
    user_loc = df_train[['userid', 'geohashed_start_loc', 'hour', 'orderid']].groupby(
        ['geohashed_start_loc', 'userid', 'hour'], as_index=False).count()
    user_loc_rank = user_loc.sort_values(['userid', 'orderid'], ascending=False)
    return user_loc_rank.groupby(['userid', 'hour'], as_index=False).nth(n)[
        ['userid',  'hour', 'geohashed_start_loc']]

def merge_get_userid_start(number,length,df_test,df_train):
    df_test['start_true'] = df_test['geohashed_start_loc']
    df_test['geohashed_start_loc'] = df_test['geohashed_start_loc'].map(lambda x: x[:length])
    df_test['hour'] = pd.to_datetime(df_test['starttime']).map(lambda x: x.strftime('%H'))
    df_test['hour'] = df_test['hour'].map(cov_2)
    del df_test['geohashed_start_loc']
    for n in range(number):
        df_test = pd.merge(df_test, get_userid_start(n,length,df_train), on=['userid', 'hour'],
                           how='left')
        df_test.rename(inplace=True, columns={'geohashed_start_loc': 'end%s' % (n)})
    df_test.rename(inplace=True, columns={'start_true': 'geohashed_start_loc'})
    print u'merge-finish'
    return df_test

##########################################################################################
# 3 时间-起点-终点
def get_start_time(n):
    print u'4 = start(7) + time', n
    df_train = pd.read_csv('../data/train.csv')
    df_train['hour'] = pd.to_datetime(df_train['starttime']).map(lambda x: x.strftime('%H'))
    df_train['hour'] = df_train['hour'].map(hour_map_s)
    df_train['geohashed_start_loc'] = df_train['geohashed_start_loc'].map(lambda x: x[:7])
    user_loc = df_train[['hour', 'geohashed_start_loc', 'geohashed_end_loc', 'orderid']].groupby(
        ['geohashed_start_loc', 'hour', 'geohashed_end_loc'], as_index=False).count()
    user_loc_rank = user_loc.sort_values(['orderid'], ascending=False)
    return user_loc_rank.groupby(['hour', 'geohashed_start_loc'], as_index=False).nth(n)[
        ['hour', 'geohashed_end_loc', 'geohashed_start_loc']]
# 根据时间段合并用户对应的地点
def merge_get_start_time(number):
    df_test = pd.read_csv('../data/test.csv')
    df_test['hour'] = pd.to_datetime(df_test['starttime']).map(lambda x: x.strftime('%H'))
    df_test['hour'] = df_test['hour'].map(hour_map_s)
    df_test['start_true'] = df_test['geohashed_start_loc']
    df_test['geohashed_start_loc'] = df_test['geohashed_start_loc'].map(lambda x: x[:7])
    for n in range(number):
        df_test = pd.merge(df_test, get_start_time(n), on=['hour','geohashed_start_loc'],
                           how='left')
        df_test.rename(inplace=True, columns={'geohashed_end_loc': 'end%s' % (n)})
    del df_test['geohashed_start_loc']
    return df_test

#########################################################
# 4 获取与用户相关的地点
def get_user_loc(n,type):
    print u'2 == user + hour + hot_top',n
    df_train = pd.read_csv('../data/train.csv')
    df_train['hour'] = pd.to_datetime(df_train['starttime']).map(lambda x: x.strftime('%H'))
    if type == 1:
        df_train['hour'] = df_train['hour'].map(hour_map)
    else:
        df_train['hour'] = df_train['hour'].map(hour_map_all)
    user_loc = df_train[['userid','hour','geohashed_end_loc','orderid']].groupby(['userid','geohashed_end_loc','hour'],as_index=False).count()
    user_loc_rank = user_loc.sort_values(['userid','orderid'],ascending=False)
    return user_loc_rank.groupby(['userid','hour'],as_index=False).nth(n)[['userid','geohashed_end_loc','hour']]
def merage_user_loc(number,type):
    df_test = pd.read_csv('../data/test.csv')
    df_test['hour'] = pd.to_datetime(df_test['starttime']).map(lambda x: x.strftime('%H'))
    if type == 1:
        df_test['hour'] = df_test['hour'].map(hour_map)
    else:
        df_test['hour'] = df_test['hour'].map(hour_map_all)
    for n in range(number):
        df_test = pd.merge(df_test,get_user_loc(n,type),on=['userid','hour'],how='left')
        df_test.rename(inplace=True,columns={'geohashed_end_loc':'end%s'%(n)})
    return df_test

###############################################
def get_hot_loc():
    print u'fifth = start '
    df_train = pd.read_csv('../data/train.csv')
    hot_loc = df_train[['geohashed_start_loc','geohashed_end_loc','orderid']].groupby(['geohashed_start_loc','geohashed_end_loc'],as_index=False).count()
    hot_loc = hot_loc.sort_values(['geohashed_start_loc','orderid'],ascending=False)
    hot_loc = hot_loc.groupby(['geohashed_start_loc'],as_index=False).nth(0)
    hot_loc.rename(inplace=True,columns={'geohashed_end_loc':'end4'})
    return hot_loc[['geohashed_start_loc','end4']]
###############################################
def get_hot_loc_hour(n):
    print u'forth = hour + start '
    df_train = pd.read_csv('../data/train.csv')
    df_train['hour'] = pd.to_datetime(df_train['starttime']).map(lambda x: x.strftime('%H'))
    df_train['hour'] = df_train['hour'].map(hour_map)
    df_train['geohashed_start_loc'] = df_train['geohashed_start_loc'].map(lambda x: x[:3])

    hot_loc_hour = df_train[['geohashed_start_loc','geohashed_end_loc','orderid','hour']].groupby(['geohashed_start_loc','geohashed_end_loc','hour'],as_index=False).count()
    hot_loc_hour = hot_loc_hour.sort_values(['geohashed_start_loc','orderid'],ascending=False)
    hot_loc_hour = hot_loc_hour.groupby(['geohashed_start_loc'],as_index=False).nth(n)
    hot_loc_hour.rename(inplace=True,columns={'geohashed_end_loc':'hour_end'})
    return hot_loc_hour[['geohashed_start_loc','hour_end','hour']]

def merge_hot_loc_test(number):
    df_test = pd.read_csv('../data/test.csv')
    df_test['hour'] = pd.to_datetime(df_test['starttime']).map(lambda x: x.strftime('%H'))
    df_test['hour'] = df_test['hour'].map(hour_map)
    df_test['s'] = df_test['geohashed_start_loc']
    df_test['geohashed_start_loc'] = df_test['geohashed_start_loc'].map(lambda x: x[:3])
    df_test = pd.merge(df_test, get_hot_loc_hour(number), on=['geohashed_start_loc', 'hour'], how='left')
    df_test.rename(inplace=True,columns={'hour_end':'hour_end4'})
    del df_test['geohashed_start_loc']
    df_test.rename(inplace=True, columns={'s': 'geohashed_start_loc'})
    return df_test[['orderid','hour_end4']]
###############################################

def get_end2start_hot_hout_loc():
    print u'get_end2start_hot_hout_loc'
    df_train = pd.read_csv('../data/train.csv')
    df_train['hour'] = pd.to_datetime(df_train['starttime']).map(lambda x: x.strftime('%H'))
    df_train['hour'] = df_train['hour'].map(hour_map)
    hot_loc = df_train[['geohashed_start_loc', 'geohashed_end_loc', 'orderid','hour']].groupby(
        ['geohashed_start_loc', 'geohashed_end_loc','hour'], as_index=False).count()
    hot_loc = hot_loc.sort_values(['geohashed_end_loc', 'orderid'], ascending=False)
    hot_loc = hot_loc.groupby(['geohashed_end_loc'], as_index=False).nth(0)
    hot_loc.rename(inplace=True, columns={'geohashed_start_loc': 'hour_end5','geohashed_end_loc':'geohashed_start_loc'})
    return hot_loc[['geohashed_start_loc', 'hour_end5','hour']]

def merge_end2start_hour():
    print u'read_test'
    df_test = pd.read_csv('../data/test.csv')
    df_test['hour'] = pd.to_datetime(df_test['starttime']).map(lambda x: x.strftime('%H'))
    df_test['hour'] = df_test['hour'].map(hour_map)
    df_test = pd.merge(df_test, get_end2start_hot_hout_loc(), on=['geohashed_start_loc', 'hour'], how='left')
    return df_test[['hour_end5']]

def get_end2start_hot_loc():
    print u'get_end2start_hot_loc'
    df_train = pd.read_csv('../data/train.csv')
    hot_loc = df_train[['geohashed_start_loc', 'geohashed_end_loc', 'orderid']].groupby(
        ['geohashed_start_loc', 'geohashed_end_loc'], as_index=False).count()
    hot_loc = hot_loc.sort_values(['geohashed_end_loc', 'orderid'], ascending=False)
    hot_loc = hot_loc.groupby(['geohashed_end_loc'], as_index=False).nth(4)
    hot_loc.rename(inplace=True, columns={'geohashed_start_loc': 'end5','geohashed_end_loc':'geohashed_start_loc'})
    return hot_loc[['geohashed_start_loc', 'end5']]

def cacl_nan(df):
    # 决策树预测的结果
    tt = pd.read_csv('../20170627_1.csv')
    print "fill"
    df['end0'] = df['end0'].fillna(tt['loc'])
    df['end1'] = df['end1'].fillna(tt['loc'])
    df['end2'] = df['end2'].fillna(tt['loc'])
    return df[['orderid','geohashed_start_loc','end0','end1','end2']]


def rmove_same(df_test):
    df_test.loc[df_test.end0 == df_test.geohashed_start_loc, 'end0'] = df_test.loc[
        df_test.end0 == df_test.geohashed_start_loc, 'end1']
    df_test.loc[df_test.end1 == df_test.geohashed_start_loc, 'end1'] = df_test.loc[
        df_test.end1 == df_test.geohashed_start_loc, 'end2']
    df_test.loc[df_test.end2 == df_test.geohashed_start_loc, 'end2'] = np.nan
    # debug
    df_test.loc[df_test.end0 == df_test.end1, 'end1'] = df_test.loc[df_test.end0 == df_test.end1, 'end2']
    df_test.loc[df_test.end0 == df_test.end2, 'end2'] = np.nan

    df_test.loc[df_test.end1 == df_test.end2, 'end2'] = np.nan
    print u'重复个数',
    print u'end0 == end1',sum(df_test['end0'] == df_test['end1']),
    print u'end0 == end2',sum(df_test['end0'] == df_test['end2']),
    print u'end1 == end2',sum(df_test['end1'] == df_test['end2']),
    print u'geohashed_start_loc == end0',sum(df_test['geohashed_start_loc'] == df_test['end0']),
    print u'geohashed_start_loc == end1',sum(df_test['geohashed_start_loc'] == df_test['end1']),
    print u'geohashed_start_loc == end2',sum(df_test['geohashed_start_loc'] == df_test['end2'])
    return df_test


def check(df1,df2):
    for end in ['end0','end1','end2']:
        print end
        df1['end0'] = df1['end0'].fillna(df2[end])
        df1['end1'] = df1['end1'].fillna(df2[end])
        df1['end2'] = df1['end2'].fillna(df2[end])
        df1 = rmove_same(df1)
    return df1

if __name__ == '__main__':
    df_test = pd.read_csv('../data/test.csv')
    df_train = pd.read_csv('../data/train.csv')
    # 用户相关的操作集合
    merge_get_userid_start_5_end_1 = merge_get_userid_start_5_end(3, 6,df_test,df_train,0)
    merge_get_userid_start_5_end_1 = merge_get_userid_start_5_end_1[['orderid','geohashed_start_loc','end0','end1','end2']]
    merge_get_userid_start_5_end_2 = merge_get_userid_start_5_end(3, 5,df_test,df_train,0)
    merge_get_userid_start_5_end_2 = merge_get_userid_start_5_end_2[['end0', 'end1', 'end2']]
    merge_get_userid_start_5_end_1 = check(merge_get_userid_start_5_end_1,merge_get_userid_start_5_end_2)

    #################################################################################
    df_test = pd.read_csv('../data/test.csv')
    df_train = pd.read_csv('../data/train.csv')
    df_train = pd.concat([df_train,df_test],axis=0)
    merge_get_userid_start = merge_get_userid_start(3,7,df_test,df_train)
    print merge_get_userid_start.head()
    merge_get_userid_start = merge_get_userid_start[['end0', 'end1', 'end2']]
    merge_get_userid_start_5_end_1 = check(merge_get_userid_start_5_end_1,merge_get_userid_start)

    #################################################################################
    # merage_user_loc_hour = merage_user_loc(3, 1)
    # merage_user_loc_hour = merage_user_loc_hour[['end0', 'end1', 'end2']]
    # merge_get_userid_start_5_end_1 = check(merge_get_userid_start_5_end_1, merage_user_loc_hour)
    #
    # merage_user_loc_all = merage_user_loc(3, 0)
    # merage_user_loc_all = merage_user_loc_all[['end0', 'end1', 'end2']]
    # merge_get_userid_start_5_end_1 = check(merge_get_userid_start_5_end_1, merage_user_loc_all)
    #################################################################################
    merge_get_start_time = merge_get_start_time(3)
    merge_get_start_time = merge_get_start_time[['end0', 'end1', 'end2']]
    merge_get_userid_start_5_end_1 = check(merge_get_userid_start_5_end_1, merge_get_start_time)

    print merge_get_userid_start_5_end_1[['orderid', 'end0', 'end1', 'end2']].dropna()

    merge_get_userid_start_5_end_1['end0'].fillna('wx4f9mu', inplace=True)
    merge_get_userid_start_5_end_1['end1'].fillna('wx4e5sw', inplace=True)
    merge_get_userid_start_5_end_1['end2'].fillna('wx4e5by', inplace=True)

    merge_get_userid_start_5_end_1[['orderid', 'geohashed_start_loc', 'end0', 'end1', 'end2']].to_csv(
        './20170725_w.csv', index=None)
    merge_get_userid_start_5_end_1[['orderid','end0','end1','end2']].to_csv('./syp.csv',index=None,header=None)
    print merge_get_userid_start_5_end_1





