"""
Review of Week-1:
on Pythontutor
Variables
Data types

Review of Week-3:
on PyCharm
Loops (for & while)
Functions

Need for more practice:
https://www.w3resource.com/python-exercises/

This week:
A) Getting input from the user (string, integer, double)
B) More complex functions with return values
C) Reading and writing into files (excel, txt/csv)
D) Regular expressions

"""

# GETTING INPUT FROM THE USER (STRING, INTEGER, DOUBLE...)
# We use the input function in Python 3. This function was raw_input() in Python 2.

# kullanıcıdan değer alır.
x1 = float ( input ( "x1: " ) )
print ( x1 )

y1 = float ( input ( "y1: " ) )
print ( y1 )


# MORE COMPLEX FUNCTIONS & THE USE OF RETURN VALUES
# iki tane girdi değerinin ortalamasını alıp döndürür.
def average(x , y):
    return (x + y) / 2
print ( average ( 4 , 3 ) )


x1 = float ( input ( "x1: " ) )
y1 = float ( input ( "y1: " ) )

def get_avg(x2 , y2):
    x2 == x1
    y2 == y1
    return (x2 + y2) / 2

print ( get_avg ( x1 , y1 ) )
# TASK-1: Find the slope of the line between 2 given points. Write two functions for this that will print and return the values respectively.

# First, please print it

# Now, please return it
x1 = float ( input ( "x1: " ) )
print ( x1 )
y1 = float ( input ( "y1: " ) )
print ( x1 )
x2 = float ( input ( "x2: " ) )
print ( x2 )
y2 = float ( input ( "y2: " ) )
print ( y2 )

def slope(x1 , x2 , y1 , y2):
    return ((y1 - y2) / (x1 - x2))
print ( slope ( x1 , y1 , x2 , y2 ) )

# TASK 2
# 2 nokta arasındaki uzaklığı bulma
# iki nokta arasındaki uzaklık formülü |AB|2= (y2-y1)2 (x2 x1)2 **AB noktasını bulmak için en son karekok alınır.math.sqrt karekök alır.
import math

# TASK-2: Please write the following calculation as a function and return the calculated value (distance) instead of printing it.
# Please get the input from the user while doing this calculation.
x1 = float ( input ( "x1: " ) )
y1 = float ( input ( "y1: " ) )
x2 = float ( input ( "x2: " ) )
y2 = float ( input ( "y2: " ) )

def calculateDistance(x1 , y1 , x2 , y2):
    dist = math.sqrt ( (x2 - x1) ** 2 + (y2 - y1) ** 2 )
    return dist
print ( calculateDistance ( x1 , y1 , x2 , y2 ) )

var1 , var2 = input ( "enter two numbers:" ).split ( ',' )

# TASK-3: Find the Max of three numbers.
# def max_of_two( x, y ): if x > y: return x return y
# def max_of_three( x, y, z ): return max_of_two( x, max_of_two( y, z ) ) print(max_of_three(3, 6, -5)

num1 = float ( input ( "num1: " ) )
num2 = float ( input ( "num2: " ) )


def bigger(x , y):
    x = num1
    y = num2
    if x > y:
        return x
    return y


print ( bigger ( num1 , num2 ) )

num1 = float ( input ( "num1: " ) )
num2 = float ( input ( "num2: " ) )
num3 = float ( input ( "num3: " ) )


def biggest(x , y , z):
    x = num1
    y = num2
    z = num3
    if z > bigger ( x , y ):
        return z
    return bigger ( x , y )

print ( biggest ( num1 , num2 , num3 ) )

# Write a Python function to create and print a list where the values are square of numbers between 1 and 20 (both included).
def printValues():
    l = list ()
    for i in range ( 1 , 21 ):
        l.append ( i ** 0.5 )
    print ( l )


printValues ()

# https://www.w3resource.com/python-exercises/python-functions-exercise-1.php
# https://www.w3resource.com/python-exercises/python-functions-exercise-2.php
# https://www.w3resource.com/python-exercises/python-functions-exercise-16.php
# READING THE FILES AND WRITING INTO THEM
# Here is the sample file: text.txt

"""What is Python language?
Python is a widely used high-level, general-purpose, interpreted, dynamic programming language.
Its design philosophy emphasizes code readability, 
and its syntax allows programmers to express concepts in fewer lines of code than possible in
languages such as C++ or Java.
Python supports multiple programming paradigms,
 including object-oriented, imperative and functional programming or procedural styles. 
It features a dynamic type system and automatic memory management and has a large and comprehensive standard library.
The best way we learn anything is by practice and exercise questions.
 We  have started this section for those (beginner to intermediate) who are
familiar with Python."""

# check whether a file exists.
import os.path
import os

titles_file = os.path.join("BDA507-Labwork-W5.txt")
titles = [line.rstrip() for line in open('BDA507-Labwork-W5.txt')]

open ( 'BDA507-Labwork-W5.txt')
print ( os.path.isfile ( 'BDA507-Labwork-W5' ) )


# FILE I/O
def file_read(fname):
    txt = open ( 'BDA507-Labwork-W5.txt' )
    print ( txt.readline () )


file_read ( 'BDA507-Labwork-W5.txt' )


# READING FIRST N LINES OF A FILE

def file_read_from_head(fname , nlines):
    from itertools import islice
    with open ( fname ) as f:
        for line in islice ( f , nlines ):
            print ( line )


file_read_from_head ( 'BDA507-Labwork-W5.txt' , 3 )


# Please append text to a file and display the text.

def file_read(fname):
    from itertools import islice
    with open ( fname , "w" ) as myfile:
        myfile.write ( "Python Exercises\n" )
        myfile.write ( "Java Exercises" )
    txt = open ( fname )
    print ( txt.read () )


file_read ( 'BDA507-Labwork-W5.txt' )


# Read a file line by line and store it into a list.

def file_read(fname):
    with open ( fname ) as f:
        # Content_list is the list that contains the read lines.
        content_list = f.readlines ()
        print ( content_list )


file_read ( 'BDA507-Labwork-W5.txt' )


# Read a file line by line store it into a variable.

def file_read(fname):
    with open ( fname , "r" ) as myfile:
        data = myfile.readlines ()
        print ( data )


file_read ( 'BDA507-Labwork-W5.txt' )


# Please count the number of lines in a text file.

def file_lengthy(fname):
    with open ( fname ) as f:
        for i , l in enumerate ( f ):
            pass
    return i + 1


print ( "Number of lines in the file: " , file_lengthy ( "BDA507-Labwork-W5.txt" ) )

# Please count the frequency of words in a file.

from collections import Counter


def word_count(fname):
    with open ( fname ) as f:
        return Counter ( f.read ().split () )


print ( "Number of words in the file :" , word_count ( "deneme.txt" ) )

# Write a Python program to write a list content to a file.
# Array pyhton'da listedir.
color = ['Red' , 'Green' , 'White' , 'Black' , 'Pink' , 'Yellow']
with open ( 'abc.txt' , "w" ) as myfile:
    for c in color:
        myfile.write ( "%s\\" % c )
content = open ( 'abc.txt' )
print ( content.read () )

# bir listenin ilk elemanını yazdırmak için aşağıdaki yöntem kullanılabilir.
print ( color[1] )

# 3 üncü eleman dahil değildir.
# 0 default olduğundan yazılmasa da olur.
# Array'in ilk 3 elemanını yazıdırı, 3 hariç.
print ( color[0:3] )

# EXP# function give distance of 2 points.
x1 = int ( ("x1: ") )
print ( x1 )
x2 = int ( ("x2: ") )
print ( x1 )
y1 = int ( ("y1: ") )
print ( x2 )
y2 = int ( ("y2: ") )
print ( y2 )

import math

p1 = [4 , 0]
p2 = [6 , 6]


def distance(x11 , x22 , y11 , y22):
    x11 == x1
    x22 == x2
    y11 = y1
    y22 = y2
    distance = math.sqrt ( ((x11[0] - x22[0]) ** 2) + ((y11[1] - y22[1]) ** 2) )
    distance = (((x11[0] - x22[0]) ** 2) + ((y11[1] - y22[1]) ** 2)) ** 0.5
    return distance


# PYTHON REGULAR EXPRESSIONS
# https://www.w3resource.com/python-exercises/re/index.php

# Check that a string contains only a certain set of characters (in this case a-z, A-Z and 0-9)
import re


# re read , w ise write
def is_allowed_specific_char(string):
    charRe = re.compile ( r'[^a-zA-Z0-9.]' )
    string = charRe.search ( string )
    return not bool ( string )


print ( is_allowed_specific_char ( "ABCDEFabcdef123450" ) )
print ( is_allowed_specific_char ( "*&%@#!}{" ) )

# Check if a given string that has an a followed by zero or more b's. 'ab*?'
# What about a given string that has an a followed by one or more b's? We use this: 'ab+?'
# What about a given string that has an a followed by zero or only one b? We use this: 'ab??'
# What about a given string that has an a followed by 3 b's? We use curly brackets and put the relevant number inside the parentheses: 'ab{3}?'
# What about a a given string that has an a followed by 2 to 3 b's? We use curly brackets and put the relevant number inside the parentheses: 'ab{2,3}?'
# What about an 'a' followed by anything, ending in 'b'. We use this expression 'a.*?b$'


import re


def text_match(text):
    patterns = 'a.*?b$'
    if re.search ( patterns , text ):
        return 'Found a match!'
    else:
        return ('Not matched!')


print ( text_match ( "ac" ) )
print ( text_match ( "abc" ) )
print ( text_match ( "abbcb" ) )

# Find the following strings:
# matches a word at the beginning of a string.'^\w+'
# matches a word at the end of a string. '\w+\S*$'
# matches a word at end of string, with optional punctuation. '\w+\S*$'
# matches a word containing 'z'. '\w*z.\w*'
# matches a word containing 'z', not start or end of the word. '\Bz\B'
# match a string that contains only upper and lowercase letters, numbers, and underscores. '^[a-zA-Z0-9_]*$'

import re


def text_match(text):
    patterns = '\w+\S*$'
    if re.search ( patterns , text ):
        return 'Found a match!'
    else:
        return ('Not matched!')


print ( text_match ( "The quick brown fox jumps over the lazy dog." ) )
print ( text_match ( " The quick brown fox jumps over the lazy dog. " ) )

# Check whether a string will start with a specific number. "^5"
# Check for a number at the end of a string. ".*[0-9]$"

import re


def match_num(string):
    text = re.compile ( r"^5" )
    if text.match ( string ):
        return True
    else:
        return False


print ( match_num ( '5-2345861' ) )
print ( match_num ( '6-2345861' ) )

# Replace whitespaces with an underscore and vice versa.

import re

text = 'Python Exercises'
text = text.replace ( " " , "_" )
print ( text )
text = text.replace ( "_" , " " )
print ( text )

# Extract year, month and date from an url.

import re


def extract_date(url):
    return re.findall ( r'/(\d{4})/(\d{1,2})/(\d{1,2})/' , url )


url1 = "https://www.washingtonpost.com/news/football-insider/wp/2016/09/02/odell-beckhams-fame-rests-on-one-stupid-little-ball-josh-norman-tells-author/"
print ( extract_date ( url1 ) )

# Convert a date of yyyy-mm-dd format to dd-mm-yyyy format.

import re


def change_date_format(dt):
    return re.sub ( r'(\d{4})-(\d{1,2})-(\d{1,2})' , '\\3-\\2-\\1' , dt )


dt1 = "2026-01-02"
print ( "Original date in YYY-MM-DD Format: " , dt1 )
print ( "New date in DD-MM-YYYY Format: " , change_date_format ( dt1 ) )

# Now let's learn how to plot figures in python
# Please draw a line with suitable label in the x axis, y axis and a title.
# File, seetings project interpreter'dan yüklenilmek istenen oyhton paketleri yüklenebilir.
import matplotlib.pyplot as plt

X = range ( 1 , 50 )
Y = [value * 3 for value in X]
print ( "Values of X:" )
print ( *range ( 1 , 50 ) )
print ( "Values of Y (thrice of X):" )
print ( Y )
# Plot lines and/or markers to the Axes.
# x ve y değerlerini plot eder.
plt.plot ( X , Y )
# Set the x axis label of the current axis.
plt.xlabel ( 'x - axis' )
# Set the y axis label of the current axis.
plt.ylabel ( 'y - axis' )
# Set a title
plt.title ( 'Draw a line.' )
# Display the figure.
# show ile grafi gösterilir.
plt.show ()

# Please draw a line using given axis values with suitable label in the x axis , y axis and a title.

import matplotlib.pyplot as plt

# x axis values
x = [1 , 2 , 3]
# y axis values
y = [2 , 4 , 1]
# Plot lines and/or markers to the Axes.
plt.plot ( x , y )
# Set the x axis label of the current axis.
plt.xlabel ( 'x - axis' )
# Set the y axis label of the current axis.
plt.ylabel ( 'y - axis' )
# Set a title
plt.title ( 'Sample graph!' )
# Display a figure.
plt.show ()

# Please draw a line using given axis values with suitable label in the x axis , y axis and a title from the given file:
# testplot.txt
# bir dosyadan data oknup oradan da grafik düzenlenebilir.
import matplotlib.pyplot as plt

with open ( "testplot.txt" ) as f:
    data = f.read ()
data = data.split ( '\n' )
x = [row.split ( ' ' )[0] for row in data]
y = [row.split ( ' ' )[1] for row in data]
plt.plot ( x , y )
# Set the x axis label of the current axis.
plt.xlabel ( 'x - axis' )
# Set the y axis label of the current axis.
plt.ylabel ( 'y - axis' )
# Set a title
plt.title ( 'Sample graph!' )
# Display a figure.
plt.show ()

# Please draw line charts of the financial data of Alphabet Inc. between October 3, 2016 to October 7, 2016.
# The sample financial data is in fdata.csv:

"""import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('fdata.csv', index_col=0)
df.plot(x=df.index, y=df.columns)
plt.show()
print(df.index)
print(df.columns)"""
# Please plot two or more lines on same plot with suitable legends of each line.
# plt alias'ı daha sonra kullanılabilir.

import matplotlib.pyplot as plt

# line 1 points
x1 = [10 , 20 , 30]
y1 = [20 , 40 , 10]
# plotting the line 1 points
plt.plot ( x1 , y1 , label="line 1" )
# line 2 points
x2 = [10 , 20 , 30]
y2 = [40 , 10 , 30]
# plotting the line 2 points
plt.plot ( x2 , y2 , label="line 2" )

x3 = [10 , 20 , 30]
y3 = [15 , 15 , 15]
plt.plot ( x3 , y3 , label="line3" )

plt.xlabel ( 'x - axis' )
# Set the y axis label of the current axis.
plt.ylabel ( 'y - axis' )
# Set a title of the current axes.
plt.title ( 'Two or more lines on same plot with suitable legends ' )
# show a legend on the plot
plt.legend ()
# Display a figure.
plt.show ()

# Please plot two or more lines with legends, different widths and colors.
#  is an alias plt
import matplotlib.pyplot as plt

# line 1 points
import matplotlib.pyplot as plt

x = range ( -10 , 10 )
y = [value ** 2 for value in X]
z = [value ** 3 for value in X]
print ( "Values of x:" )
print ( *range ( -10 , 10 ) )
print ( "Values of Y (thrice of X):" )
print ( y )
print ( "Values of Z (thrice of X):" )
print ( z )
plt.title ( 'Two or more lines with different widths and colors with suitable legends ' )
# Display the figure.
plt.plot ( x , color='blue' , linewidth=3 , label='line1-width-3' )
plt.plot ( y , color='red' , linewidth=5 , label='line2-width-5' )
plt.plot ( z , color='red' , linewidth=5 , label='line2-width-5' )
# Set the x axis label of the current axis.
plt.xlabel ( 'x - axis' )
# Set the y axis label of the current axis.
plt.ylabel ( 'y - axis' )
# Set a title
plt.title ( 'Draw a line.' )
# Display the figure.
plt.show ()

# TASK: Please insert another line that should in green. You can decide the numbers of the line.
# x,x'2,x'3 farklı renkler
# isimleri linear,scurve
# küp şeklinde gösterecek
# ...
# ...
# ...
import matplotlib.pyplot as plt

# line 1 points
x1 = [-50 , 20 , 10]
y1 = [50 , 20 , 10]
# line 2 points
x2 = [10 , 20 , 30]
y2 = [40 , 10 , 30]
# line 2 points
z1 = [10 , 20 , 30]
z2 = [40 , 10 , 30]
# Set the x axis label of the current axis.
plt.xlabel ( 'x - axis' )
# Set the y axis label of the current axis.
plt.ylabel ( 'y - axis' )
# Set a title
plt.title ( 'Two or more lines with different widths and colors with suitable legends ' )
# Display the figure.
plt.plot ( x1 , y1 , color='blue' , linewidth=3 , label='line1-width-3' )
plt.plot ( x2 , y2 , color='green' , linewidth=3 , label='line2-width-5' )
# show a legend on the plot
plt.legend ()
plt.show ()
