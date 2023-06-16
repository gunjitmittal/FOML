import pandas as pd
import numpy as np

# Loading the data and dropping the index axis
df = pd.read_csv('archive/kidney_disease.csv')
df  = df.drop(['id'],axis=1)


# separating data into different classes
real = ['sc','pot','hemo','rc',]
integer = ['age','bp','bgr','bu','sod','pcv','wc',]
label = ['classification']
cat = list(set(df.columns) - set(real)-set(integer)-set(label))

# Removing parsing errors
df = df.replace('\t?',np.nan)
df = df.replace('\tyes','yes')
df = df.replace(' yes','yes')
df = df.replace('yes\t','yes')
df = df.replace('\tno','no')
df = df.replace('ckd\t','ckd')
df = df.replace('ckd',1)
df = df.replace('notckd',0)


# Filling the null values with mean you can also use other statistic like mode or median
for r in real:
    mean = np.array(df[r][~df[r].isna()]).astype('float').mean()
    df[r] = df[r].fillna(mean)
for i in integer:
    mean = np.array(df[i][~df[i].isna()]).astype('int').mean()
    df[i] = df[i].fillna(int(mean))


X = df.drop(label,axis=1)
Y = df[label]

# You need to convert the catagorical variables to binary u can use pd.get_dummies to do so