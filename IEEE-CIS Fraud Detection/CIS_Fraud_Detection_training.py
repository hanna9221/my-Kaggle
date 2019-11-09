import numpy as np
import pandas as pd
import time
import xgboost as xgb
import lightgbm as lgb

input_dir = '..input/'
SEED      = 42

data_train = pd.read_csv(input_dir+'data_train.csv', index_col='TransactionID')
Y_train = pd.read_csv(input_dir+'label_train.csv', index_col='TransactionID')['isFraud']
X_test = pd.read_csv(input_dir+'X_test.csv', index_col='TransactionID')

data_all = data_train.append(X_test, verify_integrity=True)
del data_train, X_test
print('Data loaded.')
# =============================================================================
def reduce_mem_usage(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

data_all = reduce_mem_usage(data_all)
# =============================================================================
# Label encoding

cate_cols = ['month', 'day', 'weekday', 'hour', 'day_hour', 'weekday_hour',
             'P_email', 'R_email', 'P_emaildomain', 'R_emaildomain',
             'card_cmb', 'card1_card2', 'card1_card3', 'card1_card5', 
             'card4_card6', 'card1_addr1', 'card1_addr2', 'card2_addr1', 
             'card2_addr2', 'card3_addr1', 'addr1_addr2', 'card1_dist1', 
             'card1_dist2', 'card4_dist1', 'addr1', 'addr2',
             'Amt_decimal_len', 'Amt_interval', 'ProductCD',
              'DeviceType', 'DeviceInfo', 'DeviceBrand', 'UserAgent']
cate_cols = cate_cols + [f'card{i}' for i in range(1,7)] + \
                        [f'M{i}' for i in range(1,10)] + \
                        [f'id_{i}' for i in range(12,39)]
for col in cate_cols:
    data_all[col] = pd.factorize(data_all[col])[0]

# Columns to be dropped.
drop_cols = ['TransactionDT', 'Date']
data_all.drop(drop_cols, axis=1, inplace=True)
# =============================================================================
def split_df(df):
    return df.iloc[:590540], df.iloc[590540:]
X_train, X_test = split_df(data_all)
del data_all
print(X_train.shape)
print(X_test.shape)
# =============================================================================
# Training
print('Training session start.')
start = time.time()

# XGBoost
clf = xgb.XGBClassifier(learning_rate=0.02,
                        n_estimators=600,
                        objective='binary:logistic',
                        eval_metric='auc',
                        n_jobs=8,
                        max_depth=9,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_alpha=0.1,
                        reg_lambda=0.9,
                        verbosity=1,
                        random_state=SEED)

# LGBM
#params={'learning_rate': 0.01,
#        'n_estimators': 1200,
#        'objective': 'binary',  
#        'metric': 'auc',
#        'num_leaves': 352,
#        'n_jobs': 8,
#        'random_state': SEED,
#        'max_depth': -1,
#        'subsample': 0.8,
#        'colsample_bytree': 0.8,
#        'reg_alpha': 0.1,
#        'reg_lambda': 0.01
#       }
#clf = lgb.LGBMClassifier(**params)
#clf.fit(X_train, Y_train)


#split = [0, 590540//2, 590540]
#for i in range(2):
#    clf.fit(X_train.iloc[split[i]:split[i+1]], Y_train.iloc[split[i]:split[i+1]])


sample = pd.read_csv('input/sample_submission.csv')
sample['isFraud'] = clf.predict_proba(X_test)[:,1]
sample.to_csv('sub.csv', index=False)

end = time.time()
print('time spent =', round(end - start, 2), 's')
print('End.')
