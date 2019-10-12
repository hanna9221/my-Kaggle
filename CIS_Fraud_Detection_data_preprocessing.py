import numpy as np
import pandas as pd
import datetime

input_dir = '..input/'

train_transaction = pd.read_csv(input_dir+'train_transaction.csv', 
                                index_col='TransactionID')
train_identity = pd.read_csv(input_dir+'train_identity.csv', 
                             index_col='TransactionID')
test_transaction = pd.read_csv(input_dir+'test_transaction.csv', 
                               index_col='TransactionID')
test_identity = pd.read_csv(input_dir+'test_identity.csv', 
                            index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', 
                                left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', 
                              left_index=True, right_index=True)

del train_transaction, train_identity
del test_transaction, test_identity

label_train = train['isFraud']
train.drop('isFraud', axis=1, inplace=True)
#data_all = pd.concat([train, test])
data_all = train.append(test, verify_integrity=True)
del train, test
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
#train = reduce_mem_usage(train)
# =============================================================================
# Make some new features!!!
# DeviceInfo & UserAgent
# ***UserAgent
def UserAgent(string):
    if string==string:
        if string[:6] == 'Build/':
            return string[6:]
        part = string.split(' Build/')
        if len(part) > 1:
            return part[1]
    return 'unknown'
data_all['UserAgent'] = data_all['DeviceInfo'].map(UserAgent)

def cleanUA(string):
    if string==string:
        if string[:6] == 'Build/':
            return 'unknown'
        part = string.split(' Build/')
        if len(part) > 1:
            return part[0]
        return string
    return 'unknown'
data_all['DeviceInfo'] = data_all['DeviceInfo'].map(cleanUA)

# ***DeviceBrand
data_all['DeviceBrand'] = np.nan
DI = data_all['DeviceInfo']
na = DI[DI.isnull()].index
data_all['DeviceBrand'][na] = 'unknown'

def isSony(s):
    if s==s:
        if len(s) != 5:
            return False
        elif s[0] in ['C', 'D', 'E', 'F', 'G', 'H']:
            for c in s[1:]:
                if ord(c) < 48 or ord(c) > 57:
                    return False
            return True
        return False
    else:
        return False
data_all['isSony'] = data_all['DeviceInfo'].map(isSony)
idx = data_all[data_all['isSony']].index
data_all['DeviceBrand'][idx] = 'Sony'
data_all.drop('isSony', axis=1, inplace=True)

def isAlcatel(s):
    if s==s:
        if len(s) == 5 and s[4] in ['A','B','E','F','G','I','M','N','O','S','T','W','X','Z']:
            for c in s[:4]:
                if ord(c) < 48 or ord(c) > 57:
                    return False
            return True
        elif len(s) == 6 and s[0]=='A':
            for c in s[1:4]:
                if ord(c) < 48 or ord(c) > 57:
                    return False
            for c in s[4:6]:
                if ord(c) < 65 or ord(c) > 90:
                    return False
            return True
        else:
            return False
    return False
data_all['isAlcatel'] = data_all['DeviceInfo'].map(isAlcatel)
idx = data_all[data_all['isAlcatel']].index
data_all['DeviceBrand'][idx] = 'Alcatel'
data_all.drop('isAlcatel', axis=1, inplace=True)


brand_dict = {'rv|Android|unknown|en-|es-': 'unknown', 'Linux': 'Linux',
              'MacOS': 'MacOS', 'iOS Device|iPhone': 'iOS Device',
              'Windows|Microsoft|Trident|Win64|IE9': 'Windows',
              'M4': 'M4', 'Z9': 'Qmobile', 'NX': 'Nextbook',
              'F80 PIABELLA': 'F2_Mobile', 'verykool': 'verykool',
              'Z410|B1-|A3-|A1-850': 'Acer', 'Le|LEX722': 'LeEco',
              'ONEPLUS|ONE|A0001': 'ONEPLUS', 'Neffos': 'Neffos',
              'Nokia|NOKIA|Lumia|TA-': 'Nokia', 'AERIAL': 'AERIAL',
              'MDDRJS': 'Dell', 'ASUS|P00|K01': 'ASUS', 'RCT': 'RCA',
              'KF': 'Amazon', 'CPH|R8|A1601|A37f': 'OPPO', 'Vivo|vivo': 'Vivo',
              'Aquaris': 'Aquaris', 'iris|LAVA|A3_mini|Grand2c': 'LAVA',
              'AW790': 'AIWA', 'BB|STV100': 'BlackBerry', 'ZUUM': 'ZUUM',
              'TR10': 'ECS', 'TREKKER': 'Crosscall', 'ZA': 'Zonda',
              'S60': 'CAT', 'S60 Lite': 'Doogee', 'BV': 'Blackview',
              'BLU|Energy|STUDIO|Grand': 'BLU', 'NATIVO': 'NATIVO', 'FP': 'FP',
              'Alumini|ALUMINI': 'K&S', 'Hisense': 'Hisense',
              'PULP|TOMMY2|LITE': 'Wiko', 'Ilium|ILIUM|Alpha': 'Lanix', 
              'QTA|QMV7A': 'Verizon', 'AX6|AX8|AX9|AX10': 'Bmobile', 
              'C67|KYY22|E6790TM': 'Kyocera', 'SOV|SO-|LT22i|LT30p|SGP': 'Sony',
              'Mi |MI |Redmi': 'Xiaomi', 'Lenovo|Tab|MOT-|YOGA|TAB7': 'Lenovo',
              'Polaroid|Turbo C5|P50|PSP|P552|P4526A|PMID|M10': 'Polaroid',
              'HT|2PS64|0PM92|2PYB2|2PZC5|0PAJ5': 'HTC',
              'HUAWEI|hi62|-L23|-L01|-L02|-L03|-L09|-AL|A-A|B3-A|T1-A21w|-U0|-U2|-TL\
               |-W|P10|-LX|-L1|-L2|-L3|-L5|BGO-DL09': 'Huawei',
              'ZTE|BLADE|Blade|Z55|Z79|Z81|Z85|Z83|Z91|Z95|N817\
               |Z96|Z97|Z98|N9560|N951|N913|K90U|NX5|K88': 'ZTE',
              'VS4|VS5|VS8|VS9|LM|LG|L-03K|VK4|VK7|VK8|RS9': 'LG',
              'Moto |moto |Moto|moto|XT1|XT8': 'Motorola', 
              'Nexus|Pixel': 'Google', 'SH-': 'others', 
              'VS5012': 'Vulcan', 'Infinit|INFINIT|Tmovi': 'Timovi',
              'SAMSUNG|Galaxy|gxq|SC|GT-|SM-|A10 |SGH|SPH|S9\+|Note8': 'SAMSUNG',
              'ALCATEL|Alcatel|ONE TOUCH|8062|808|A462C|REVVLPLUS': 'Alcatel'
              }

brand_dict_pre = {'MI': 'Xiaomi', 'Mi': 'Xiaomi', 'POCOPHONE': 'Xiaomi',
                  'M4': 'M4', 'Shift': 'Shift', 'Tab8': 'Acer',
                  'K10': 'unknown', 'Max10': 'unknown',
                  'S8': 'SAMSUNG', 'A10': 'SAMSUNG', 'Gravity': 'Gravity',
                  '7_Plus': 'iOS Device', '8_Plus': 'iOS Device',
                  'SKY_5.0LM': 'others', 'Minion_Tab': 'others',
                  'HT07': 'others', 'NYX_A1': 'others', 'MAMI': 'others',
                  'VFD': 'others', 'DT': 'others', 'AM508': 'others',
                  'S.N.O.W.4': 'others', 'T1': 'others'}

def replace(search, brand_name):
    DI = data_all['DeviceInfo']
    brand = DI[DI.str.contains(search, na=False)]
    data_all['DeviceBrand'][brand.index] = brand_name
    print(brand_name, brand.shape)

def brand(brand_dict, brand_dict_pre=None):
    for key in brand_dict.keys():
        replace(key, brand_dict[key])
    if brand_dict_pre:
        DI = data_all['DeviceInfo']
        for key in brand_dict_pre.keys():
            brand = DI[DI==key]
            data_all['DeviceBrand'][brand.index] = brand_dict_pre[key]
            print(brand_dict_pre[key], brand.shape)
brand(brand_dict, brand_dict_pre)

DB = data_all['DeviceBrand']
brand_na = DB[DB.isnull()].index
data_all['DeviceBrand'][brand_na] = 'others'
print('DeviceInfo & UserAgent are done.')
# =============================================================================
# NaN
data_all['nan'] = data_all.isnull().sum(axis=1)

# C1-C14
data_all['C1/C6'] = data_all.apply(lambda row: round((row.C1+1)/(row.C6+1),3), axis=1)
data_all['C2/C1'] = data_all.apply(lambda row: round((row.C2+1)/(row.C1+1),3), axis=1)
data_all['C5/C1'] = data_all.apply(lambda row: round((row.C5+1)/(row.C1+1),3), axis=1)
data_all['C5/C9'] = data_all.apply(lambda row: round((row.C5+1)/(row.C9+1),3), axis=1)
data_all['C11/C14'] = data_all.apply(lambda row: round((row.C11+1)/(row.C14+1),3), axis=1)
data_all['C1-C6'] = data_all.apply(lambda row: row.C1-row.C6, axis=1)
data_all['C2-C1'] = data_all.apply(lambda row: row.C2-row.C1, axis=1)
print('C cols are done.')

# D1-D15
data_all['D1-D2'] = data_all.apply(lambda row: row.D1-row.D2, axis=1)
data_all['D1-D3'] = data_all.apply(lambda row: row.D1-row.D3, axis=1)
data_all['D4-D15'] = data_all.apply(lambda row: row.D4-row.D15, axis=1)
print('D cols are done.')

# M1-M9
def count_M(row):
    cols = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
    temp = 0
    for col in cols:
        if row[col]=='T':
            temp += 1
    return temp
data_all['count_M'] = data_all.apply(count_M, axis=1)

# V1-V339
vcols = [f'V{i}' for i in range(1,340)]
data_all[vcols] = data_all[vcols].fillna(-1)
print('M & V cols are done.')

# date
START_DATE = '2017-11-30'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
data_all['Date'] = data_all['TransactionDT'].map(lambda x: (startdate + datetime.timedelta(seconds=x)))
data_all['month'] = data_all['Date'].dt.month.astype(str)
data_all['day'] = data_all['Date'].dt.day.astype(str)
data_all['weekday'] = data_all['Date'].dt.dayofweek.astype(str)
data_all['hour'] = data_all['Date'].dt.hour.astype(str)
data_all['day_hour'] = data_all['day'] + data_all['hour']
data_all['weekday_hour'] = data_all['weekday'] + data_all['hour']
cnt_day = data_all['day'].value_counts()
cnt_day = cnt_day / cnt_day.mean()
data_all['_count_rate'] = data_all['day'].map(cnt_day.to_dict())
print('date cols are done.')

# email
def email(s):
    if type(s) != type(''): return 'unknown'
    table = {'yahoo': 'yahoo', 'ymail': 'yahoo', 'rocketmail': 'yahoo', 'frontier': 'yahoo',
             'hotmail': 'MS', 'outlook': 'MS', 'live': 'MS', 'msn': 'MS',
             'icloud': 'apple', 'mac': 'apple', 'me.com': 'apple',
             'prodigy': 'ATT', 'att.net': 'ATT', 'sbcglobal': 'ATT', 'bellsouth': 'ATT',
             'centurylink': 'CL', 'embarqmail': 'CL', 'q.com': 'CL',
             'aol.com': 'Aol', 'aim.com': 'Aol', 'verizon': 'Aol', 
             'twc.com': 'Charter', 'charter': 'Charter', 'roadrunner': 'Charter', 'rr.com': 'Charter',
             'comcast': 'comcast', 'gmail': 'gmail', 'anonymous': 'anonymous', 
             'proton': 'proton', 'cox.net': 'cox.net', 'optonline.net': 'optonline.net'}
    for k in table.keys():
        if k in str(s):
            return table[k]
    return 'others'
data_all['P_email'] = data_all['P_emaildomain'].map(email)
data_all['R_email'] = data_all['R_emaildomain'].map(email)
print('date & email cols are done.')

# card & addr & dist
data_all['card_cmb'] = data_all['card1'].astype(str) + '_' + data_all['card2'].astype(str) + '_' + \
                       data_all['card3'].astype(str) + '_' + data_all['card5'].astype(str)
data_all['card1_card2'] = data_all['card1'].astype(str) + '_' + data_all['card2'].astype(str)
data_all['card1_card3'] = data_all['card1'].astype(str) + '_' + data_all['card3'].astype(str)
data_all['card1_card5'] = data_all['card1'].astype(str) + '_' + data_all['card5'].astype(str)
data_all['card4_card6'] = data_all['card4'].astype(str) + '_' + data_all['card6'].astype(str)
data_all['card1_addr1'] = data_all['card1'].astype(str) + '_' + data_all['addr1'].astype(str)
data_all['card1_addr2'] = data_all['card1'].astype(str) + '_' + data_all['addr2'].astype(str)
data_all['card2_addr1'] = data_all['card2'].astype(str) + '_' + data_all['addr1'].astype(str)
data_all['card2_addr2'] = data_all['card2'].astype(str) + '_' + data_all['addr2'].astype(str)
data_all['card3_addr1'] = data_all['card3'].astype(str) + '_' + data_all['addr1'].astype(str)
data_all['addr1_addr2'] = data_all['addr1'].astype(str) + '_' + data_all['addr2'].astype(str)
data_all['card1_dist1'] = data_all['card1'].astype(str) + '_' + data_all['dist1'].astype(str)
data_all['card1_dist2'] = data_all['card1'].astype(str) + '_' + data_all['dist2'].astype(str)
data_all['card4_dist1'] = data_all['card4'].astype(str) + '_' + data_all['dist1'].astype(str)


# Amt
def Amt_decimal_len(amount):
    split = str(amount).split('.')
    if len(split) > 1:
        return len(split[-1])
    return 0
data_all['Amt_decimal_len'] = data_all['TransactionAmt'].map(Amt_decimal_len)
data_all['Amt_decimal'] = ((data_all['TransactionAmt'] - data_all['TransactionAmt'].astype(int))*1000).astype(int)
data_all['Amt_interval'] = pd.qcut(data_all['TransactionAmt'], 20)

cols = ['ProductCD','card1','card2','card5','card6', 'addr1', 'P_email', 'R_email']
for f in cols:
    data_all[f'Amt_mean_{f}'] = data_all.groupby([f])['TransactionAmt'].transform('mean')
    data_all[f'Amt_std_{f}'] = data_all.groupby([f])['TransactionAmt'].transform('std')
    data_all[f'Amt_pct_{f}'] = (data_all['TransactionAmt'] - data_all[f'Amt_mean_{f}']) / data_all[f'Amt_std_{f}']
print('Amt cols are done.')

# frequency encoding
cols = ['ProductCD', 'card1', 'card2', 'card3', 'card5', 'card4_card6', 
        'addr1', 'addr2', 'addr1_addr2']
cols += [f'C{i}' for i in range(1,15)]
for f in cols:
    vc = data_all[f].value_counts(dropna=False)
    data_all[f'count_{f}'] = data_all[f].map(vc)

for f in ['card1', 'card2', 'card3', 'card5', 'card4_card6', 'addr1']:
    vc1 = data_all[data_all['C5']==0][f].value_counts(dropna=False)
    vc2 = data_all[data_all['C5']!=0][f].value_counts(dropna=False)
    data_all[f'C5_frac_{f}'] = data_all[f].map(vc2) / (data_all[f].map(vc1)+0.01)
print('frequency encoding is done.')
# =============================================================================
def split_df(df):
    return df.iloc[:590540], df.iloc[590540:]
data_train, X_test = split_df(data_all)
del data_all
print(data_train.shape)
print(X_test.shape)

# Save data
data_train.to_csv('data_train.csv')
label_train = pd.DataFrame(label_train)
label_train = label_train.rename(columns={0: 'isFraud'})
label_train.to_csv('label_train.csv', index_label='TransactionID')
X_test.to_csv('X_test.csv')
print('Data is saved.')