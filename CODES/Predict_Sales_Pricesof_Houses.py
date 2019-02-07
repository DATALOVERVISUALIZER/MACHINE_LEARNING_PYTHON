" HOUSE PRICES COMPREHENSIVE DATA EXPLORATION WITH PYHTON "
" COMPREHENSIVE DATA EXPLORATION FOR HOUSE PRICES "
"  LEYLA YİĞİT  "
"  MEF UNIVERSITY"
"JANUARY 2019 TERM PROJECT "

import pandas as pd #Analysis
import numpy as np #Analysis
from scipy.stats import norm #Analysis
from sklearn.preprocessing import StandardScaler #Analysis
from scipy import stats #Analysis
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import warnings
warnings.filterwarnings('ignore')
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
import gc

##############DATA ANALYSIS##############

# #CUSTOM COLOR SEQUENCE CREATE
# import numpy as np
# import matplotlib.pyplot as plt

#
# # Have colormaps separated into categories:
# # http://matplotlib.org/examples/color/colormaps_reference.html
# cmaps = [
#     # ('Perceptually Uniform Sequential', [
#     #         'viridis', 'plasma', 'inferno', 'magma']),
#          ('Sequential', [
#             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
#          # ('Sequential (2)', [
#          #    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
#          #    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
#          #    'hot', 'afmhot', 'gist_heat', 'copper']),
#          # ('Diverging', [
#          #    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
#          #    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
#          # ('Qualitative', [
#          #    'Pastel1', 'Pastel2', 'Paired', 'Accent',
#          #    'Dark2', 'Set1', 'Set2', 'Set3',
#          #    'tab10', 'tab20', 'tab20b', 'tab20c']),
#          # ('Miscellaneous', [
#          #    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
#          #    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
#          #    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])
# ]
#
#
# nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
# gradient = np.linspace(0, 1, 256)
# gradient = np.vstack((gradient, gradient))
#
#
# def plot_color_gradients(cmap_category, cmap_list, nrows):
#     fig, axes = plt.subplots(nrows=nrows)
#     fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
#     axes[0].set_title(cmap_category + ' colormaps', fontsize=14)
#
#     for ax, name in zip(axes, cmap_list):
#         ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
#         pos = list(ax.get_position().bounds)
#         x_text = pos[0] - 0.01
#         y_text = pos[1] + pos[3]/2.
#         fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
#
#     # Turn off *all* ticks & spines, not just the ones with colormaps.
#     for ax in axes:
#         ax.set_axis_off()
#
#
# for cmap_category, cmap_list in cmaps:
#     plot_color_gradients(cmap_category, cmap_list, nrows)
#
# plt.show()




#get data
df_train = pd.read_csv('C:/Users/Kafein/PycharmProjects/Introduction to Python/Class/CLASS_FINAL_WORKS/FINAL/train.csv')
df_test = pd.read_csv('C:/Users/Kafein/PycharmProjects/Introduction to Python/Class/CLASS_FINAL_WORKS/FINAL/test.csv')
df_sample_submission = pd.read_csv('C:/Users/Kafein/PycharmProjects/Introduction to Python/Class/CLASS_FINAL_WORKS/FINAL/sample_submission.csv')

print(df_train.tail())
print(df_test.tail())
print(df_sample_submission.tail())
print("train.csv. Shape: ",df_train.shape)
print("test.csv. Shape: ",df_test.shape)
print("sample_submission.csv. Shape: ",df_sample_submission.shape)

#data types
print(df_train.dtypes)
print(df_sample_submission.dtypes)

#In order to understand our data
#descriptive statistics summary
df_train['SalePrice'].describe()

#Dealing with Missing and Abnormal Variables
df_train.dropna()  #deleting the nan variables
print(df_train.shape)  #checking shape of the data again
df_test.dropna()  #deleting the nan variables
print(df_test.shape)  #checking shape of the data again



#Is there any dublicate ID
A=df_train.duplicated('Id')
print(sum(i for i in A if i == True))  #check any dublicate id

'''Finding Missing values'''
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


#missing data histogram
#histogram
#missing_data = missing_data.head(20)
percent_data = percent.head(20)
percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Value (%)", fontsize = 20)


#outliers
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# OverallQual : 4
# OverallQual : 8s
# OverallQual : 10


df_train[df_train['OverallQual'] == 4][df_train['SalePrice'] > 200000]
df_train = df_train[df_train['Id'] != 458]
df_train[df_train['OverallQual'] == 8][df_train['SalePrice'] > 500000]
df_train[df_train['OverallQual'] == 10][df_train['SalePrice'] < 180000]


var = 'SaleCondition'
data = pd.concat([df_train[df_train['OverallQual'] == 10]['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)


df_train[df_train['OverallQual'] == 4][df_train['SalePrice'] > 200000] #outlier 4 , we see from the graph that outlier 4 bigger than 200000
df_train = df_train[df_train['Id'] != 458]
df_train[df_train['OverallQual'] == 8][df_train['SalePrice'] > 500000]#outlier 8
df_train[df_train['OverallQual'] == 10][df_train['SalePrice'] < 180000]#outlier 10

#remove all the outliers
df_train = df_train[df_train['Id'] != 524][df_train['Id'] != 1299]
var = 'Neighborhood'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)






#histogram
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(df_train['SalePrice'])

#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())








# VARIABLES RELATIONSHIP WITH EACH OTHER
#SalePrice correlation matrix (zoomed heatmap style)

#saleprice correlation matrix
k = 10 #number of variables for heatmap
corrmat = df_train.corr(method='spearman') # correlation
cols = corrmat.nlargest(k, 'SalePrice').index # nlargest : Return this many descending sorted values
cm = np.corrcoef(df_train[cols].values.T) # correlation
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(8, 6))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


'''9 most relevant variables with SalePrice
•	OverallQual : Overall material and finish quality
•	GrLivArea : Above grade (ground : the portion of a home that is above the ground) living area square feet
•	GarageCars : Size of garage in car capacity
•	GarageArea : Size of garage in square feet
•	TotalBsmtSF : Total square feet of basement area
•	1stFlrSF : First Floor square feet
•	FullBath : Full bathrooms above grade
•	TotRmsAbvGrd : Total rooms above grade (does not include bathrooms)
•	YearBuilt : Original construction date'''


#scatter plot grlivarea/saleprice

"""3. The scatterplot
The scatterplot helps us visualize the relationship between two or more variables. The code needed to execute a
scatterplot is shown below:"""
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# Relationship with categorical features

# Relationship between categorical data
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.title("relationship year and sales")
fig.patch.set_facecolor('grey') #change colors
plt.xticks(rotation=90);

#TRY DİFFERENT GRAPH
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

x = df_train['SalePrice']
y = df_train['YearBuilt']
colors =df_train['YearBuilt']
sizes = df_train['SalePrice']

plt.scatter(x, y, c=colors, s=sizes, alpha=0.2,
            cmap='viridis')
plt.colorbar();
plt.title("relationship year and sales")
fig.patch.set_facecolor('dark') #change colors
  # show color scale


#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,cmap='bwr')
plt.title("Correlation Analysis Between Variables");

#scatterplot
import seaborn as sns; sns.set(style="ticks", color_codes=True);sns.set(rc={'axes.facecolor':'Snow', 'figure.facecolor':'Gray'})
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols],size = 2.5)


var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
x = df_train['SalePrice']
y = df_train['YearBuilt']

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
       sns.set_palette("husl")
sinplot()


import numpy as np
import matplotlib.pyplot as plt

var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
x = df_train['YearBuilt']
y = df_train['SalePrice']
# plt.plot(x,  color='blue')
plt.bar(x, y,color='Blue')
plt.xlabel('YearBuilt')
plt.ylabel('SalePrice')
plt.title('Change sales by year')
plt.show()


