###TASKS
#TASK-1 and TASK-2
import math
import numpy as np
x = np.arange(1, 20)
print("Array:",x)
print(len(x))
i=0
v_max=x[0]
v_min=x[0]
v_total=x[0]
v_mean=0
v_sum = 0
while i < len(x)-1:
    if v_min < x[i+1]:
        v_max = x[i+1]
    else:
        v_min = x[i+1]
    v_total += x[i + 1]
    i = i + 1
    v_mean = v_total/len(x)
print("Mean:",v_mean)

print(len(x))

i=0
while i < len(x)-1:
    v_sum += (x[i] - v_mean)**2
    i += 1
print("Max:",v_max)
print("Min:",v_min)
print("Mean:",v_mean)
print("StDev:",math.sqrt(v_sum/len(x)-1))






#TASK-3
def bubbleSort(num_arr):
    len_arr = len(num_arr)
    for x in range(len_arr):
        for y in range(0, len_arr - x - 1):
            if num_arr[y] > num_arr[y + 1]:
                num_arr[y], num_arr[y + 1] = num_arr[y + 1], num_arr[y]
    return num_arr


num_array = [1,3,5,7,9,2,4,6,8]
print(bubbleSort(num_array))


#TASK-4
def selectionSort(num_arr):
   for x in range(len(num_arr)-1,0,-1):
       ind_max=0
       for y in range(1,x+1):
           if num_arr[y]>num_arr[ind_max]:
               ind_max = y
       temp = num_arr[x]
       num_arr[x] = num_arr[ind_max]
       num_arr[ind_max] = temp
   return num_arr

### BDA507 WEEK-11 Fall 2018-19

"""
How do you import data into Python?
http://lineardata.net/how-do-you-import-data-into-python/

Python is increasingly growing in popularity thanks to the large number of packages that cater to the need of the data
scientist. Importing data into Python thus becomes the starting point for any data science project that you will
undertake. This guide gives you a comprehensive introduction into the world of importing data into Python.

There are number of file formats that are available that offer you with a source of structured and unstructured data.

The various sources of structured data are:

.CSV files
.TXT files
Excel files
SAS and STATA files
HDF5 files
Matlab files
The various sources of unstructured data are

Data from the web in the form of HTML pages
This guide will teach you the fundamentals of importing data from all these sources straight into your python
workspace of choice with minimal effort. So let’s get coding!"""


"""CSV files usually contain mixed data types and it’s best to import the same as a data frame using the pandas package in
python. You can do this with the code snippet shown below:"""

#Importing the csv file
import pandas as pd
filename = 'deneme.csv'
data = pd.read_csv(filename)
data.head()


"""TXT files
The next type of file that we might encounter on our quest to becoming a master data scientist is the .TXT file.
Importing these files into python is as easy as importing the CSV file and can be done with the code snippet shown below:"""

#Importing the txt file:
with open("file.txt", 'r') as myfile:
    print(myfile.read())

"""The above line of code that uses the ‘with’ is called as the context manager in python. The open() function opens the
file – ‘file.txt’ as a read only document using the argument ‘r’. We then read the file using the myfile.read() argument
and printing out the same. If you want to edit the .txt file that you just imported you would want to use the
‘w’ argument with the open() function instead of the ‘r’ argument."""


"""EXCEL Files
Excel files are a huge part of any business operation and it becomes imperative that you learn exactly how to import
these into python for data analysis as a pro data scientist. In order to do this we can use the code snippet shown below:"""

# Importing excel files into Rodeo

import pandas as pd
import excel

file = "deneme.xlsx"
myfile = pd.ExcelFile(file)
print(myfile.sheet_names)  # printing out all the sheet names in the excel file
dataframe = myfile.parse('sheet1')  # extracting data from the first sheet as a dataframe

"""In the above code we first imported pandas. We then stored in the excel file into a variable called ‘file’ after which
we imported the file into python using the pd.ExcelFile() function. Using the ‘.sheet_names) we printed out the sheet
names present in the excel file. We then extracted the contents of the first sheet as a dataframe using the ‘.parse()’
function."""


"""SAS and STATA files
Statistical analytic software is widespread in the business analytics space and needs to be given due diligence.
Let’s take a close look at how can get them into python for further analysis."""

# Importing SAS files

import pandas as pd
from sas7bdat import SAS7BDAT

with SAS7BDAT("sas.sas7bdat") as myfile:
    dataframe = myfile.to_data_frame()

# Importing STATA files

import pandas as pd
stata_file = pd.read_stata("stata.dta")


"""HDF5
The HDF5 file format stands for  Hierarchal  Data Format version 5. The HDF5 is very popular for storing large
quantities of numerical data which can span from a few Gigs to exabytes. It’s very popular in the scientific community
for storing experimental data. Fortunately for us we can import these files quite easily into python by using the code
snippet shown below:"""

# Importing HDF5 files
import h5py
filename = 'myfile.hdf5'
file = h5py.File(filename, 'r')

"""MATLAB files
Matlab files are used quite extensively by electronic and computer engineers for designing various electrical and
electronic systems. Matlab is built around linear algebra and can store a lot of numerical data that we could use for
analysis.  In order to import a matlab file we can use the code snippet illustrated below:"""

# importing matlab files
import scipy.io
file = "matlab.mat"
myfile = scipy.io.loadmat(file)


"""Data from the web
Data from the web is usually in the form of unstructured data that has no order to linearity to it. However, we can find
structured data on some websites like Kaggle and the UCI machine learning repository. Such files can be downloaded
directly into python from the web using the code snippet below:"""

# Downloading CSV files from the web
from urllib.request import urlretrieve
website = "http://imkevinjolly.com/blog/dataset.csv"
urlretrieve(website, "dataset.csv")

import pandas as pd
df = pd.read_csv("dataset.csv")

"""In the code above we have used the urlretrieve package from urllib.request in order to download a csv file from the
website. We then saved it as a dataframe locally using the pandas package.

In order to import HTML pages into python we can make use of the ‘requests’ package and a couple of lines of code
that’s shown below:"""

# Importing webpages

import requests
url = "http://www.mef.edu.tr"
file = requests.get(url)
word_file = file.text

""" The requests.get() function sends a request to the server to import the webpage while the file.text will convert
the webpage into a text file.

Most of the time data from webpages don’ code that does not resonate well with anybody. In order to make sense of the data that we import from the web
we have t really make a lot of sense. It’s usually in the form of jumbled up text and
a lot ofto make use of the BeautifulSoup package that is offered by Python."""

# Using BeautifulSoup

from bs4 import BeautifulSoup
import requests

url = "http://www.mef.edu.tr"
file = requests.get(url)
word_file = file.text
Beauty = BeautifulSoup(word_file)
print(Beauty.prettify())
print(Beauty.title)



"""Basics of Data Cleaning:
http://lineardata.net/the-ultimate-guide-to-cleaning-data-in-python/

Happiness inside a Job:
https://www.kaggle.com/harriken/clean-data-python/notebook
https://github.com/Guillem-db/Happiness-inside-a-job-PyDataBcn17

Taser Data:
https://trendct.org/2016/08/05/real-world-data-cleanup-with-python-and-pandas/
https://qxf2.com/blog/cleaning-data-python-pandas/
https://github.com/KarrieK/pandas_data_cleaning
https://www.dataquest.io/blog/data-cleaning-with-python/
https://ssds.stanford.edu/python-tutorials-cleaning-and-scraping-data
https://s3.amazonaws.com/assets.datacamp.com/production/course_3485/slides/ch5_slides.pdf
https://stackoverflow.com/questions/13867294/cleaning-big-data-using-python
https://www.kaggle.com/harriken/clean-data-python/notebook"""


"""
DATA CLEANING
A data scientist roughly spends about 70% of his time cleaning messy and unstructured data. If you are going to be
spending this much time with one task you might as well know how to clean data well. This guide gives you a
comprehensive step by step procedure when it comes to cleaning your data in Python.

http://lineardata.net/the-ultimate-guide-to-cleaning-data-in-python/
"""
import pandas as pd
import numpy as np
import seaborn as sns #visualization da iyidir.
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

iris = pd.read_csv("iris.csv")
iris.columns
iris.head()

iris.head()
iris.tail()
"""This will show you the first 5 rows and the last 5 rows of your dataset so that you can briefly inspect your dataset.  
Note: The ‘df’ stands for a dataframe. We saved our dataset into a dataframe called ‘df’ using pandas prior to this –
Please read our http://lineardata.net/how-do-you-import-data-into-python/ guide to learn how you can import data into python."""

"""This returns a list of all the column names that you could inspect. 
Now assume I have a column named “Work_tym” – We can obviously see that time is spelled wrong as “tym”. In order to change this we have to use the code shown below:"""

iris["variety_new"] = iris["variety"]
iris.head()

iris = iris.drop("variety_new", 1)

iris.head()

"""After this we want to check if all the columns have the right data type associated with it. For example, sometimes a
column that has numeric data might be stored as an ‘object’ type because it has a missing value stored as ‘missing’.
To address this issue we need to first:"""

iris.info()

"""With this, we can see what types each of the columns are. We can now change the type of any particular column using
the code below that The code above converts the ‘float64’ type of the satisfaction_level column to ‘category’."""

iris['sepal.length'] = iris['sepal.length'].astype('category')

iris.shape

iris.info()

iris.head()

iris['sepal.length'] = iris['sepal.length'].astype('float64')

iris.info()

"""which returns the number of columns and rows in the datasets.

Another useful metric you could add to your initial exploration would be the frequency counts of each element in a
particular column. We can implement this using the code shown below:"""

iris.variety.value_counts()

iris["variety"].value_counts()

iris["sepal.length"].value_counts()

"""STEP 2: Visualize your data
I’m not asking you to visualize your data in great detail but I want you to understand if there are errors in terms of outliers. There are 3 fundamental visualizations that you must carry out
1.The Histogram:
The histogram gives you a basic frequency count and allows you to determine weather a particular column is normally distributed or not. The code to plot a histogram is shown below:"""

import matplotlib.pylot as plt
df.time_spend_company.plot('hist')
plt.show()

"""2. The boxplot
A box plot is the best way to figure out if you have outliers in your dataset. It gives you the values of the 1st, 2nd and 3rd quartiles as well the median.
 In order to plot a box plot we need to follow the code below:"""

import matplotlib.pyplot as plt

df.boxplot(column='time_spend_company', by='left')
plt.show()

"""3. The scatterplot
The scatterplot helps us visualize the relationship between two or more variables. The code needed to execute a
scatterplot is shown below:"""

import matplotlib.pyplot as plt

df.plot(kind='scatter', x='time_spend_company', y='satisfaction_level')
plt.show()

"""STEP 3: Deal with your missing values
Dealing with missing values is crucial because of it does not make sense to feed these missing values into a machine
learning algorithm or a neural network. There are a couple of ways you could deal with missing values. 
The first method is to drop them completely from the entire dataset or from a particular column. 
The second method is to fill them with some summary statistic or a number or a string based on the problem at hand.
In order to drop all the missing value from a dataframe we use the code shown below:"""

df.dropna()

"""If you want to drop the missing value from a particular column only we can use the code shown below:"""

df.salary.dropna()

"""Note that the column we are dropping the missing values from is called ‘salary’ in the code above.
The next way that you could deal with missing values is to fill them. In the code below I have filled the missing values
in the ‘salary’ column with the value of 0."""

df['salary'] = df['salary'].fillna(0)

"""You could also fill in missing values with strings as shown below:"""

df['salary'] = df['salary'].fillna('no_value')

"""The final way you could fill missing value is to use a summary statistic like the mean. You can implement this using the
code shown below:"""

mean = df['salary'].mean()
df['salary'] = df['salary'].fillna(mean)

"""STEP 4: Merge datasets
In most of your data analysis we are going to have multiple datasets. The idea here is to follow steps 1 to 3 for each
dataset that you have understudy and once it’s all cleaned you can merge them together.
Assume we have two data frames “DF1” and “DF2” and we want to merge them we simply use the code shown below:"""

pd.merge(left = DF1, right = DF2)

"""In conclusion, following steps 1 to 4 should yield you a relatively clean dataset depending on the complexity of the
dataset in hand. There are techniques like ‘melting’ which will give you a much greater control over the data cleaning
process but this will be covered in another guide in which manipulating data frames using pandas is the key focus."""

"""TASK-2: Please work on the given data set on movies and apply the cleaning methods on this."""


"""STATISTICS

80 öğrenci matematik sınav notları
40 kadın 40 erkek öğrenci
kadınlar mı erkekler mi yoksa berabere mi?


TASK-3: Please provide a compare two-means test for the given data set.


http://www.scipy-lectures.org/packages/statistics/index.html"""



import numpy as np
from scipy import stats
np.random.seed(100)
male=np.random.randint(60,80,35,dtype="int")
female=np.random.randint(70,90,35,dtype="int")
stats.ttest_ind(male,female)
import pandas as pd
import numpy as np
import seaborn as sns #visualization da iyidir.
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')





import numpy as np
import skimage

a = np.zeros((3, 3))
a[1, 1] = 1
print(a)
gaussian(a, sigma=0.4)  # mild smoothing
gaussian(a, sigma=1)  # more smoothing
# Several modes are possible for handling boundaries
gaussian(a, sigma=1, mode='reflect')
# For RGB images, each is filtered separately
from skimage.data import astronaut
image = astronaut()
filtered_img = gaussian(image, sigma=1, multichannel=True)

f = misc.face(gray=True)  # retrieve a grayscale image
import matplotlib.pyplot as plt
plt.imshow(f, cmap=plt.cm.gray)

# Remove axes and ticks
plt.axis('off')
plt.contour(f, [50, 200])
plt.imshow(f[320:340, 510:530], cmap=plt.cm.gray)

from scipy import misc
face = misc.face(gray=True).astype(float)
blurred_f = ndimage.gaussian_filter(face, 3)
plt.imshow(blurred_f)
plt.show()




"""
========================================================================================================================
======================= Classification applications on the handwritten digits data =====================================
========================================================================================================================
In this example, you will see two different applications of Naive Bayesian Algorithm on the
digits dataset.
"""

print(__doc__)


import pylab as pl
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

########################################################################################################################
##################################### GETTING THE DATA & PREPARATIONS ##################################################
########################################################################################################################

np.random.seed(42)
digits = load_digits()  # the whole dataset with the labels and other information are extracted
data = scale(digits.data)  # the data is scaled with the use of z-score
data
n_samples, n_features = data.shape  # the no. of samples and no. of features are determined with the help of shape
n_digits = len(np.unique(digits.target))  # the number of labels are determined with the aid of unique formula
labels = digits.target  # get the ground-truth labels into the labels

print(labels)
print(digits.keys())  # this command will provide you the key elements in this dataset
print(digits.DESCR)  # to get the descriptive information about this dataset

pl.gray()
digits.images[0]
pl.matshow(digits.images[0])
pl.show()
print(digits.images[0])

########################################################################################################################
########################################################################################################################

from sklearn.model_selection import train_test_split  # some documents still include the cross-validation option but it no more exists in version 18.0
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pylab as plt
y = digits.target
y
X = digits.data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

########################################################################################################################
########################################################################################################################

gnb = GaussianNB()
fit = gnb.fit(X_train, y_train)
fit
predicted = fit.predict(X_test)
predicted
print(confusion_matrix(y_test, predicted))
print(accuracy_score(y_test, predicted)) # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print(accuracy_score(y_test, predicted, normalize=False))  # the number of correct predictions
print(len(predicted)) # number of all of the predictions

########################################################################################################################
########################################################################################################################

gnb = GaussianNB()
fit2 = gnb.fit(X, y)
predictedx = fit2.predict(X)
print(confusion_matrix(y, predictedx))
print(accuracy_score(y, predictedx)) # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print(accuracy_score(y, predictedx, normalize=False))  # the number of correct predictions
print(len(predictedx))  # number of all of the predictions

unique_y, counts_y = np.unique(y, return_counts=True)
print(unique_y, counts_y)

unique_p, counts_p = np.unique(predictedx, return_counts=True)
print(unique_p, counts_p)
print((predictedx == 0).sum())
########################################################################################################################
########################################################################################################################










