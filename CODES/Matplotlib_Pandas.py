
""" BDA507 WEEK-9 Contents:

- Guideline about Term-Projects (20%)
- Class in Python & Exercises
- Information about Libraries/Packages
    * numpy
    * matplotlib
    * pandas
    * scipy
    * seaborn
- Exercises on Pandas library
- Work on Iris Dataset
"""

"""
The basic idea behind an object-oriented programming (OOP) is to combine both data and associated procedures (known as
methods) into a single unit which operate on the data. Such a unit is called an object.Python is an object-oriented
language, everything in Python is an object. We have already worked with some objects in Python, (See Python data type
chapter) for example strings, lists are objects defined by the string and list classes which are available by default
into Python. Let's declare two objects a string and a list and test their type with type() function.

Defining a class:
In object oriented programming classes and objects are the main features. A class creates a new data type and objects
are instances of a class which follows the definition given inside the class. Here is a simple form of class definition.

class Student:
    Statement-1
    Statement-1
    ....
    ....
    ....
    Statement-n
A class definition started with the keyword 'class' followed by the name of the class and a colon.
The statements within a class definition may be function definitions, data members or other statements.
When a class definition is entered, a new namespace is created, and used as the local scope.

Creating a Class:
Here we create a simple class using class keyword followed by the class name (Student) which follows an indented block
of segments (student class, roll no., name).
#studentdetails.py
class Student:
        stu_class = 'V'
        stu_roll_no = 12
        stu_name = "David"

Class Objects:
There are two kind of operations class objects supports : attribute references and instantiation. Attribute references
use the standard syntax, obj.name for all attribute references in Python. Therefore if the class definition (add a
method in previous example) look like this

#studentdetails1.py
class Student:
    #A simple example class
    stu_class = 'V'
    stu_roll_no = 12
    stu_name = "David"
    def messg(self):
            return 'New Session will start soon.'
then Student.stu_class, Student.stu_roll_no, Student.stu_name are valid attribute reference and returns 'V', 12,
'David'. Student.messg returns a function object. In Python self is a name for the first argument of a method which is
different from ordinary function. Rather than passing the object as a parameter in a method the word self refers to the
object itself. For example if a method is defined as avg(self, x, y, z), it should be called as a.avg(x, y, z). See the
output of the attributes in Python Shell.

__init__ method:
There are many method names in Python which have special importance. A class may define a special method named __init__
which does some initialization work and serves as a constructor for the class. Like other functions or methods __init__
can take any number of arguments. The __init__ method is run as soon as an object of a class is instantiated and class
instantiation automatically invokes __init__() for the newly-created class instance. See the following example a new,
initialized instance can be obtained by:

#studentdetailsinit.py
class Student:
    # A simple example class
    def __init__(self, sclass, sroll, sname):
        self.c = sclass
        self.r = sroll
        self.n = sname
    def messg(self):
            return 'New Session will start soon.'
"""

# Write a Python program to convert an integer to a roman numeral.
class py_solution:
    def int_to_Roman(self, num): # tam sayıyı roman karaktere çevirir.
        val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4,
            1
            ]
        syb = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
            ]
        roman_num = ''
        i = 0
        while  num > 0:
            for _ in range(num // val[i]):
                print(_)
                roman_num += syb[i]
                num -= val[i]
            i += 1
        return roman_num

print(py_solution().int_to_Roman(1))
print(py_solution().int_to_Roman(4000))
print(py_solution().int_to_Roman(27))


# Write a Python program to convert a roman numeral to an integer.

class py_solution:
    def roman_to_int(self, s):
        rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        int_val = 0
        for i in range(len(s)):
            if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
                int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
            else:
                int_val += rom_val[s[i]]
        return int_val

print(py_solution().roman_to_int('MMMCMLXXXVI'))
print(py_solution().roman_to_int('MMMM'))
print(py_solution().roman_to_int('C'))


#####################################################################################################################
### TASK-1: Write a Python class named Circle constructed by a radius and two methods which will compute the area and
### the +perimeter of a circle.
#####################################################################################################################
import math
class Circle():
    def __init__(self,r):
        self.radius=r

    def area(self):
        return math.pi*math.pow(self.radius,2)

    def perimeter (self):
        return 2*math.pi*self.radius

newCircle=Circle(3)
print(newCircle.area())
print(newCircle.perimeter())


# ...
# ...
# ...
# ...

import math
#####################################################################################################################
### TASK-2: Write a Python class named Cylinder constructed by a radius and two methods which will compute the Volume and
### the SurfaceArea of a circle.
#####################################################################################################################
class Cylinder():
    def __init__(self, r,h):
        self.radius = r
        self.height = h

    def Volume(self): #yuzey alanı
        return math.pi*math.pow(self.radius,2)*self.height
#2.π.r.(r + h)
    def SurfaceArea(self):
        return 2*math.pi*self.radius*(self.radius+self.height)

NewCylinder = Cylinder(8,2)  # this initializes a circle object with radius = 8
print(NewCylinder.Volume())  # this line will calculate and print the area of the object
print(NewCylinder.SurfaceArea())

NewCircle = Circle(8)  # this initializes a circle object with radius = 8
print(NewCircle.area())  # this line will calculate and print the area of the object
print(NewCircle.perimeter())  # this line will calculate and print the perimeter of the object

#####################################################################################################################
# Please calculate the surface area and volume of a cylinder (radius and height)

#pirizmanın hacmini ve yüzey alanını bulan bir class ve obje yazınız.
class Pirizma():
    def __init__(self, h,l,w):
        self.height =h
        self.length =l
        self.width=w

    def Volume(self): #yuzey alanı
        return self.height*self.length*self.width
#2*(wl+hl+hw)
    def SurfaceArea(self):
        return 2*(self.width*self.length+self.height*self.length+self.height*self.width)

NewPirizma = Pirizma(3,4,5)  # this initializes a circle object with radius = 8
print(NewPirizma.Volume())  # this line will calculate and print the area of the object
print(NewPirizma.SurfaceArea())

#####################################################################################################################
#####################################################################################################################

# NumPy. NumPy is the fundamental package for scientific computing with Python. It provides some advance math
# functionalities to python.
# For more info: http://www.numpy.org/-- matematiksel işlemleri en hızlı yapan kütüphane

# matplotlib. matplotlib is a python 2D plotting library which produces publication quality figures in a variety of
# hardcopy formats and interactive environments across platforms.  It is a must have for any data scientist or any data
# analyst.
# For more info: http://matplotlib.org/

# pandas: Pandas is a library for operating with table-like structures. It comes with a powerful DataFrame object, which
# is a multi-dimensional array object for efficient numerical operations similar to NumPy’s ndarray with additional
# functionalities.
# For more info: http://pandas.pydata.org/

# SciPy. SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and
# engineering. It provides a wide range of algorithms and mathematical tools for data scientist.
# For more info: https://www.scipy.org/


######################################### PANDAS LIBRARY #############################################################

# Full documentation via this address: https://pandas.pydata.org/pandas-docs/stable/index.html

# PANDAS Package Overview: https://pandas.pydata.org/pandas-docs/stable/overview.html

"""pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis
tools for the Python programming language.
pandas consists of the following elements
* A set of labeled array data structures, the primary of which are Series and DataFrame
* Index objects enabling both simple axis indexing and multi-level / hierarchical axis indexing
* An integrated group by engine for aggregating and transforming data sets
* Date range generation (date_range) and custom date offsets enabling the implementation of customized frequencies
* Input/Output tools: loading tabular data from flat files (CSV, delimited, Excel 2003), and saving and loading pandas
objects from the fast and efficient PyTables/HDF5 format.
* Memory-efficient “sparse” versions of the standard data structures for storing data that is mostly missing or mostly constant (some fixed value)
* Moving window statistics (rolling mean, rolling standard deviation, etc.)"""

# Basics of Pandas: https://pandas.pydata.org/pandas-docs/stable/10min.html

################################# IRIS DATASET #################

# The following piece of code is obtained from the following internet site:
# https://www.kaggle.com/ashokdavas/iris-data-analysis-pandas-numpy

import pandas as pd
import numpy as np
import seaborn as sns #visualization da iyidir.
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

iris = pd.read_csv("iris.csv")
iris.columns
iris.head()

# univariate plots to see individual distribution
sns.distplot(a=iris["sepal.length"],rug=True) #kde=true & hist=true by default

# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature
# not univariate in complete sense. 3 different plots for 3 species values
sns.FacetGrid(iris, hue="variety", size=6) \
    .map(sns.kdeplot, "sepal"".length") \
    .add_legend()
# sns.plt.show()

# you can see that all below combinations are providing a good distribution of "Species"
sns.factorplot(x="sepal.length",y="sepal.width",data=iris,hue="variety")
sns.factorplot(x="sepal.length",y="petal.length",data=iris,hue="variety")
sns.factorplot(x="petal.width",y="sepal.width",data=iris,hue="variety")
sns.factorplot(x="petal.length",y="petal.width",data=iris,hue="variety")

# let you easily view both a joint distribution and its marginals at once.
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
# Can't provide hue in joint plot
sns.jointplot(x="sepal.length", y="sepal.width", data=iris,size=5,kind="scatter") #scatter is default kind
# sns.plt.show()

# A clear picture of distribution can be seen with pairplot. Pairplot displays distribution of data
# according to every combination.
# In pair plot, members except diagonals are joint plot
print (iris.columns)
sns.pairplot(iris,hue="variety",diag_kind="kde")
# sns.plt.show()

#####################################################################################################################
### TASK-2: Find the cases (no.) with highest and lowest petal/sepal length/width. ##################################
#####################################################################################################################
iris.groupby(["variety"])["sepal.length"].max()

index=iris["petal.length"].idxmax()
print(index)
iris["variety"].iloc[index]


#####################################################################################################################
### TASK-3: Find the mean values, standard deviation, range of petal/sepal length/width. ############################
#####################################################################################################################




#####################################################################################################################
### TASK-4: Provide a task that will take more than a second processing time ########################################
#####################################################################################################################

import time #timedan bir obje yaratıldıgında direk ona referans verior. Daha önce yaratılanla arasındaki farkı verir. YAratılan iki obje arasındaki farkı verir.
t0=time.time()

print(t0)
time.sleep(5)

print(time.time()-t0)


# 3-D Plotting of these features
########################################################################################################################
# This section below is for illustrating the dataset in 3-D regarding different features
########################################################################################################################

from mpl_toolkits.mplot3d import Axes3D  # to have 3d figures
from sklearn import datasets  # to obtain the dataset of iris
import numpy as np  # numpy library as you already know
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # to use the kmeans clustering algorithm
from sklearn.preprocessing import scale  # to be able to preprocess by scaling the data before clustering: See explanations in A2 for this

np.random.seed(5)
#centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()  # extract the dataset into iris
print(iris)
print(iris.DESCR)
X = iris.data  # the data part will be here
print(X)
y = iris.target  # the target items (class labels) will be here
data = scale(iris.data)  # use of scale function here to standardize a dataset along any axis. Center to the mean and component wise scale to unit variance
print(iris.data)  # to see the difference between the original data and scaled data lets print them
print(data)  # this is the scaled one
n_samples, n_features = data.shape  # n_samples will contain the number of samples/cases we have for the dataset we have. Here it is 150
# n_features will contain the number of features/dimensions, here, it is 4 as we mentioned: sepal_length, sepal_width, petal_length, and petal_width
no_spec = len(np.unique(iris.target)) # number of species, here we have three of them
print(y)

A = np.mean(X[0:49, 0])  # ilk bitki ilk ozellik
C = a = np.zeros(shape=(3, 4))  # np.random.randint(1, size=(3, 4))
print(C)
for i in range(n_features):
    C[0, i] = np.mean(X[0:49, i])
    C[1, i] = np.mean(X[50:99, i])
    C[2, i] = np.mean(X[100:149, i])

print(C)
labels = iris.target  # name of the classes we have (three different species)
sample_size = 150
# Below, we provide 3 different analyses with different setting as you can see below:
estimators = {'k_means_iris_3': KMeans(n_clusters=3), 'k_means_iris_5': KMeans(n_clusters=5),  # run k-means with 3 clusters
              'k_means_iris_8': KMeans(n_clusters=8)}  # run k-means with 8 clusters
###    ,  # try k-means with 8 clusters
###              'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1, init='random')}  # try k-means with 3 clusters
print(estimators)
########################################################################################################################
# This section below is for illustrating the dataset in 3-D regarding different features
########################################################################################################################
print(estimators.items())
fignum = 1
for name, est in estimators.items():  # cycle regarding the number of analysis determined above
    print(name)
    print(est)
    fig_n_clusters = est.n_clusters  # number of clusters desired in the kmeans analyses
    fig = plt.figure(fignum, figsize=(4, 3))  # figure descriptions
    plt.clf()  # clear the figure
    ax = Axes3D(fig, rect=[0, 0, .95, 1],
                elev=48,
                azim=134)  # module containing Axes3D, an object which can plot 3D objects on a 2D matplotlib figure
    plt.cla()  # clear the axis
    est.fit(X)  # fit function compute kmeans clusterimg given the array-like input, X here: training instances to cluster
    labels = est.labels_  # estimated labels
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))   # to have a 3D scatter plot we are using the 3 dimensions you can also try 1

    ax.w_xaxis.set_ticklabels([])  # Set the text values of the tick labels. Return a list of Text instances.
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')  # label the related axis
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    fignum = fignum + 1
    plt.suptitle(("TITLE = %d" % fig_n_clusters), fontsize=14, fontweight='bold')  #provide a title for the figure

fig = plt.figure(fignum, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(), X[y == label, 0].mean() + 1.5, X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w',
                        facecolor='w'))

# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3],
           X[:, 0],
           X[:, 2],
           c=y)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.suptitle(("TITLE = %d" % no_spec), fontsize=14, fontweight='bold')
plt.show()

