import pandas as pd
import matplotlib.pyplot as plt
import data_helper as helper
from sklearn.model_selection import train_test_split

classes = {
1:	'HDF',
2:	'MDF',
3:	'LDF',
4:	'YRF',
5:	'YS',
6:	'OL',
7:	'F',
8:	'E',
9:	'G',
10:	'NP',
11:	'OP',
12:	'DF',
13:	'C',
14:	'R',
15:	'RT',
16:	'W',
17:	'P',
18:	'SH',
19:	'AQ',
20:	'AG',
21:	'TP'
}


concessions = [ 'app_riau']
data_frame = helper.get_all_concession_data(concessions)
print(data_frame.head())


X = data_frame[ (data_frame.clas==21.0) | (data_frame.clas==4.0)  ] #  | (data_frame.clas==9.0)  | (data_frame.clas==18.0) | | ((data_frame.clas==12.0) | (data_frame.clas==9.0)  | (data_frame.clas==18.0)] # or data_frame.clas==6.0]# or data_frame[['class']]==4.0 or data_frame[['class']]==9.0 or data_frame[['class']]==21.0 or data_frame[['class']]==18.0]
print(X.head())
X = X.filter(['S2_nir', 'VH_2015', 'S2_swir1', 'ndvi_s2', 'clas', 'class_binary', 'S2_vape', 'median_rededge3', 'ls5_ndvi_2009_max'])
print(X.head())
myclass_data, X_test= train_test_split(X, train_size=0.0040, test_size=0.1,
                                                        random_state=13)
df = pd.DataFrame(dict(x=myclass_data.S2_swir1, y=myclass_data.median_rededge3, label=myclass_data.clas))
groups = df.groupby('label')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=classes.get(int(name)))
ax.legend()
ax.set_xlabel('S2_swir1')
ax.set_ylabel('median_rededge3')
plt.show()
#Add the dictionary values to give more meaningful designations of features.


df = pd.DataFrame(dict(x=myclass_data.S2_vape, y=myclass_data.VH_2015, label=myclass_data.clas))
groups = df.groupby('label')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=classes.get(int(name)))
ax.legend()
ax.set_xlabel('S2_vape')
ax.set_ylabel('VH_2015')
plt.show()

df = pd.DataFrame(dict(x=myclass_data.ndvi_s2, y=myclass_data.ls5_ndvi_2009_max, label=myclass_data.clas))
groups = df.groupby('label')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=classes.get(int(name)))
ax.legend()
ax.set_xlabel('ndvi_s2')
ax.set_ylabel('ls5_ndvi_2009_max')
plt.show()



df = pd.DataFrame(dict(x=myclass_data.S2_swir1, y=myclass_data.median_rededge3, label=myclass_data.class_binary))
groups = df.groupby('label')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
ax.legend()
ax.set_xlabel('S2_swir1')
ax.set_ylabel('median_rededge3')
plt.show()
#Add the dictionary values to give more meaningful designations of features.


df = pd.DataFrame(dict(x=myclass_data.S2_vape, y=myclass_data.VH_2015, label=myclass_data.class_binary))
groups = df.groupby('label')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
ax.legend()
ax.set_xlabel('S2_vape')
ax.set_ylabel('VH_2015')
plt.show()

df = pd.DataFrame(dict(x=myclass_data.ndvi_s2, y=myclass_data.ls5_ndvi_2009_max, label=myclass_data.class_binary))
groups = df.groupby('label')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
ax.legend()
ax.set_xlabel('ndvi_s2')
ax.set_ylabel('ls5_ndvi_2009_max')
plt.show()