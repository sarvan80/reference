
import re

# Define a pattern and a string
pattern = r"^hello"
string = "hello world"

# Perform a match
match = re.match(pattern, string)

match.expand(r"\g<0> there")

if match:
    print("Match found:", match.group())
else:
    print("No match")
'''
No, re.match() does not match all occurrences. It only checks for a match at the beginning of the string. If you want to match all occurrences of a pattern in a string, you should use re.findall() or re.finditer().
'''
########## Most consumate concise printout worthy ready reckoner for Regex, Strings, Date, Tips, pandas ...(stats ######

import re
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import numpy as np
import scipy as sp
import pandas as pd

# 1049 -- boilerref
# pip install -U pandasql
from pandasql import sqldf


txt = "The rain in Spain falls mainly in the plain!"
# split()--Split the string into a list, splitting it wherever the RE matches

# sub()--Find all substrings where the RE matches, and replace them with a different string

# subn()---Does the same thing as sub(), but returns the new string and the number of replacements

# match()-Determine if the RE matches at the beginning of the string.

# search()-Scan through a string, looking for any location where this RE matches.

# findall()-Find all substrings where the RE matches, and returns them as a list.

# finditer()- Find all substrings where the RE matches, and returns them as an iterator.

# querying the match object

# group()--Return the string matched by the RE
# start()--Return the starting position of the match
# end()--Return the ending position of the match
# span()--Return a tuple containing the (start, end) positions of the match


# patterns
# . ^ $ * + ? { } [ ] \ | ( ) # simple meta # all brackets,multiply subtract ^.$ ?\|
# | or ^-Begining $-End () same as math

x = re.findall("^hello", txt)
x = re.findall("planet$", txt)
x = re.findall("he..o", txt)
print(x)
x = re.findall("[a-m]", txt)
print(x)
x = re.findall("[a-m]", txt)
print(x)
x = re.findall("he.{2}o", txt)
print(x)
x = re.findall("he.*o", txt)
print(x)
x = re.findall("he.+o", txt)
print(x)
x = re.findall("he.?o", txt)
print(x)

# Patterns ready reckoner
'''

[] 	A set of characters 	"[a-m]" 	
\ 	Signals a special sequence (can also be used to escape special characters) 	"\d" 	
. 	Any character (except newline character) 	"he..o" 	
^ 	Starts with 	"^hello" 	
$ 	Ends with 	"planet$" 	
* 	Zero or more occurrences 	"he.*o" 	
+ 	One or more occurrences 	"he.+o" 	
? 	Zero or one occurrences 	"he.?o" 	
{} 	Exactly the specified number of occurrences 	"he.{2}o" 	
| 	Either or 	"falls|stays" 	
() 	Capture and group

# special patterns
# \A,\b,\B,\d,\D,\s,\S,\w,\W,\Z ABDSWZ (badasszw big A & Z)

\A 	Returns a match if the specified characters are at the beginning of the string 	"\AThe" 	
\b 	Returns a match where the specified characters are at the beginning or at the end of a word
(the "r" in the beginning is making sure that the string is being treated as a "raw string") 	r"\bain"

r"ain\b" 	

\B 	Returns a match where the specified characters are present, but NOT at the beginning (or at the end) of a word
(the "r" in the beginning is making sure that the string is being treated as a "raw string") 	r"\Bain"
r"ain\B" 	
\d 	Returns a match where the string contains digits (numbers from 0-9) 	"\d" 	
\D 	Returns a match where the string DOES NOT contain digits 	"\D" 	
\s 	Returns a match where the string contains a white space character 	"\s" 	
\S 	Returns a match where the string DOES NOT contain a white space character 	"\S" 	
\w 	Returns a match where the string contains any word characters (characters from a to Z, digits from 0-9, and the underscore _ character) 	"\w" 	
\W 	Returns a match where the string DOES NOT contain any word characters 	"\W" 	
\Z 	Returns a match if the specified characters are at the end of the string 	"Spain\Z"


# Flags
# re.A,re.S,re.I,re.M,re.U,re.X

# 	re.A	Returns only ASCII matches
#   re.S	Makes the . character match all characters (including newline character)
#   	re.M	Returns only matches at the beginning of each line
# re.NOFLAG		Specifies that no flag is set for this pattern
# re.UNICODE	re.U	Returns Unicode matches. This is default from Python 3. For Python 2: use this flag to return only Unicode matches
# re.VERBOSE	re.X	Allows whitespaces and comments inside patterns. Makes the pattern more readable

#Tip r before words 
# Without r
print("This is a newline: \n")  # \n is treated as a newline character

# With r
print(r"This is a newline: \n")  # \n is treated as two literal characters: \ and n
'''

print(re.findall("spain", txt, re.I)) #	Case-insensitive matching

x = re.findall("falls|stays", txt)
xA = re.findall("\AThe", txt)

xb = re.findall(r"\bain", txt)

print(re.findall("spain", txt, re.I))

s='spain'
se=re.search('ai',s)
print(se)


# Date and Time functions

# import the date class
from datetime import date

my_date = date(1996, 12, 11)

print("Date passed as argument is", my_date)
print(type(my_date))
today = date.today()
print("Today's date is", today)
# date object of today's date
today = date.today()
print("Current year:", today)
print("Current year:", today.year)
print("Current month:", today.month)
print("Current day:", today.day)
# Getting Datetime from timestamp
from datetime import datetime

date_time = datetime.fromtimestamp(1887639468)
print("Datetime from timestamp:", date_time)

today = date.today()

# Converting the date to the string
Str = date.isoformat(today)
print("String Representation", Str)
print(type(Str))

from datetime import time

Time = time(11, 34, 56)
print("hour =", Time)

print("hour =", Time.hour)
print("minute =", Time.minute)
print("second =", Time.second)
print("microsecond =", Time.microsecond)

from datetime import datetime

a = datetime(1999, 12, 12, 12, 12, 12)
print("timestamp =", a.timestamp())

################## Datetime calculations ##################
from datetime import datetime, timedelta

# Using current time
ini_time_for_now = datetime.now()

# printing initial_date
print("initial_date", str(ini_time_for_now))

# Calculating future dates
# for two years
future_date_after_2yrs = ini_time_for_now + timedelta(days=730)

future_date_after_2days = ini_time_for_now + timedelta(days=2)

# printing calculated future_dates
print('future_date_after_2yrs:', str(future_date_after_2yrs))
print('future_date_after_2days:', str(future_date_after_2days))
# Timedelta function demonstration
from datetime import datetime, timedelta

# Using current time
ini_time_for_now = datetime.now()

# printing initial_date
print("initial_date", str(ini_time_for_now))

# Some another datetime
new_final_time = ini_time_for_now + timedelta(days=2)

# printing new final_date
print("new_final_time", str(new_final_time))

# printing calculated past_dates
print('Time difference:', str(new_final_time - ini_time_for_now))

import datetime
#Display Weekday as a number 0-6, 0 is Sunday
i=2018
j=6
k=1
x = datetime.datetime(i, j, k)
#x=x+1
print(x.strftime("%w"))
y=x+datetime.timedelta(days=5)
print(x)
type(y)
print(y.strftime("%B"))
print(y.strftime("%m"))
print(y.strftime("%j"))
print(y.strftime("%U"))
type(y.strftime("%B"))
z=int(y.strftime("%m"))+12

################## Datetime calculations end ##################
# s[3:len(s)-3] # string slicing and innovative ways
# pattern = r"\W+" # string slicing and innovative ways
# (y+timedelta(days=2)).strftime("%B") # string slicing and innovative ways
################## strings ##################
a = "Hello, World!"
print(a[1])

# In[ ]:
# Sub-String--Like list slicing
b = "Hello, World!"
print(b[2:5])

#################
# In[ ]:

# Strip,len,lower,Upper,Replace,Split

a = " Hello, World! "
print(a.strip()) # returns "Hello, World!"

# Replace a character in a string
a = " Hello-World! "
a = a.replace('-','')
print(a.strip()) # returns "Hello, World!"

a = "Hello, World!"
print(len(a))

a = "Hello, World!"
print(a.lower())

a = "Hello, World!"
print(a.upper())

a = "Hello, World!"
print(a.replace("H", "J"))

a = "Hello, World!"
print(a.split(",")) # returns ['Hello', ' World!']

a1=list(a)
a2=dict(a1)

# Command-line String Input
print("Enter your name:")
x = input()
print("Hello, " + x)

################## strings end ##################

################ other usefull tips ############
# ### Map, Reduce & Lambda
#
# #### https://medium.com/@happymishra66/lambda-map-and-filter-in-python-4935f248593

# In[7]:

def multiply2(x):
    return x * 2

list(map(multiply2, [1, 2, 3, 4]))  # Output [2, 4, 6, 8]

# In[8]:
list_a = [1, 2, 3]
list_b = [10, 20, 30]
map(lambda x, y: x + y, list_a, list_b)  # Output: [11, 22, 33]
# ### Dict comprehension

# In[10]:

fruits = ['apple', 'mango', 'banana', 'cherry']
{f: len(f) for f in fruits}

# In[11]:

dict1 = {1: 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
# Double each value in the dictionary
double_dict1 = {k: v * 2 for (k, v) in dict1.items() if v > 3}
print(double_dict1)

# In[13]:

double_dict1['d']

################ other usefull tips end ############
# In[11]:
def t(*args,**kwargs):
    print(kwargs)

    i1=args[0]
    i2=args[1]

    s=0

    for i in i2:
        print(i2)
        s=s+i

    return s

r=t(5,[3,4,5],{'a':1,'b':2},n1="st",n2="g3")
print(r)

################ Numpy     ############
# In[11]:

# distributions in numpy
# 1. normal distribution
nm=np.random.normal(0,1,1000)
# 2. binomial distribution
np.random.binomial(10,0.5,1000)
# 3. poisson distribution
np.random.poisson(5,1000)
# 4. uniform distribution
np.random.uniform(0,1,1000)
# 5. gamma distribution
np.random.gamma(1,1,1000)
# 6. beta distribution
np.random.beta(1,1,1000)
# 7. exponential distribution
np.random.exponential(1,1000)
# 8. geometric distribution
np.random.geometric(0.5,1000)
# 9. hypergeometric distribution
np.random.hypergeometric(10,5,5,1000)
# 10. lognormal distribution
np.random.lognormal(0,1,1000)\
# 11. multinomial distribution
np.random.multinomial(10,[0.2,0.3,0.5],1000)

# numpy to get the probability of a die roll
from numpy.random import binomial
binomial(10, .5, 1000).mean()

# distributions in scipy
# 1. normal distribution
sp.stats.norm.rvs(0,1,1000)
# 2. binomial distribution
sp.stats.binom.rvs(10,0.5,1000)
# 3. poisson distribution
sp.stats.poisson.rvs(5,1000)
# 4. uniform distribution
sp.stats.uniform.rvs(0,1,1000)
# 5. gamma distribution
sp.stats.gamma.rvs(1,1,1000)
# 6. beta distribution
sp.stats.beta.rvs(1,1,1000)
# 7. exponential distribution
sp.stats.expon.rvs(1,1000)

# setting up a experiment of coin tossing
def coin_tossing_experiment():
    return np.random.binomial(1,0.5,1000)
def coin_tossing_experiment_mean():
    return np.mean(coin_tossing_experiment())
def coin_tossing_experiment_variance():
    return np.var(coin_tossing_experiment())
def coin_tossing_experiment_mean_list():
    return [coin_tossing_experiment_mean() for i in range(1000)]
def coin_tossing_experiment_variance_list():
    return [coin_tossing_experiment_variance() for i in range(1000)]

ct=coin_tossing_experiment_mean_list()

# 1. create a pandas dataframe from a numpy array
df=pd.DataFrame({'a':np.random.normal(0,1,1000),'b':np.random.normal(0,1,1000)})

# 1. create a numpy array
na=np.array([[1,2],[3,4]])

# 2. convert the pandas dataframe to a numpy array
na=df.values

a=np.arange(10)
print(a)

# create a numpy boolean array
nb=np.full((3,3),True,dtype=bool)
print(nb)

# numpy extraction --important
ne=a[a%2==1]
print(ne)
# Replace an element that satisfies a condition with another value in numpy array
a[a%2==1]=-1
print(a)
# reshape a numpy array
a=np.arange(9)
a.reshape(3,-1)
# common items between two numpy arrays
a=np.array([1,2,3,2,3,4,3,4,5,6])
b=np.array([7,2,10,2,7,4,9,4,9,8])
c=np.intersect1d(a,b)
print(c)
np.unique(a)

# how to remove from one array those items that exist in another
a=np.array([1,2,3,4,5])
b=np.array([5,6,7,8,9])
c=np.setdiff1d(a,b)
print(c)
# get the positions where elements of two arrays match
a=np.array([1,2,3,2,3,4,3,4,5,6])
b=np.array([7,2,10,2,7,4,9,4,9,8])
c=np.where(a==b)
print(c)
# extract all numbers between a given range from a numpy array
a=np.array([2,6,1,9,10,3,27])
index=np.where((a>=5)&(a<=10))
print(index)
# how to swap two columns in a 2d numpy array
arr=np.arange(9).reshape(3,3)
print(arr)
arr[:,[1,0,2]]

# convert the function maxx that works on two scalars, to work on two arrays -- very important
def maxx(x,y):
    if x>=y:
        return x
    else:
        return y
# pair_max=np.vectorize(maxx,otypes=[float])
pair_max=np.vectorize(maxx)
m=pair_max([[5.3,7,9,8],[3,4,5,6]],[[6.4,3,4,5],[1,2,3,4]])
print(m)
# how to swap two rows in a 2d numpy array
arr=np.arange(9).reshape(3,3)
print(arr)
arr[[1,0,2],:]
# reverse the rows of a 2d array
arr=np.arange(9).reshape(3,3)  # important
print(arr)
arr[::-1]
# reverse the columns of a 2d array
arr=np.arange(9).reshape(3,3)
print(arr)
arr[:,::-1]
# create a 2d array of shape 5x3 to contain random decimal numbers between 5 and 10
arr=np.random.uniform(5,10,size=(5,3))
print(arr)

#generate a numpy array of random integers between 5 and 10
np.random.randint(5,10,size=(5,3))
# generate a numpy matrix of 5x3 random numbers sampled from a uniform distribution
np.random.uniform(5,10,size=(5,3))
# generate a random numpy 1 d integer array
np.random.randint(5,10,size=5)
# generate a random int array between 5 and 10 of size 5
np.random.randint(5,100,size=5)
# how to print only 3 decimal places in python numpy array
rand_arr=np.random.random((5,3))
print(rand_arr)
np.set_printoptions(precision=3)
print(rand_arr)
# how to pretty print a numpy array by suppressing the scientific notation
np.random.seed(100)
rand_arr=np.random.random([3,3])/1e3
print(rand_arr)
np.set_printoptions(suppress=True,precision=6)
print(rand_arr)
# how to limit the number of items printed in output of numpy array
a=np.arange(15)
print(a)
np.set_printoptions(threshold=6)
print(a)
# how to pretty print a numpy array by suppressing the scientific notation
np.random.seed(100)
rand_arr=np.random.random([3,3])/1e3
print(rand_arr)
np.set_printoptions(suppress=True,precision=6)
print(rand_arr)

url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# how to extract a particular column from 1d array of tuples    --important
iris=np.genfromtxt(url,delimiter=',',dtype=None)
species=np.array([row[4] for row in iris])
species[:5]

url1='https://raw.githubusercontent.com/Sketchjar/MachineLearningHD/main/iris.csv'
ir1=np.genfromtxt(url1,delimiter=',',dtype=None)

ir2=ir1[1:,:4]
names=['sepallength','sepalwidth','petallength','petalwidth','species']
ir2[:,:4]=ir2[:,:4].astype('float')
irispd2=pd.DataFrame(ir2)

# how to import a dataset with numbers and texts keeping the text intact in python numpy --important

iris=np.genfromtxt(url,delimiter=',',dtype='object')
names=['sepallength','sepalwidth','petallength','petalwidth','species']
iris[:3]
type(names)
print(iris)
# getting the first column of a numpy 2d array
iris[:,0]
# getting the first 4 columns of a numpy 2d array
iris[:,:4]
# convert the first column to a float array and change in place
iris[:,:4]=iris[:,:4].astype('float') # very important
print(iris)
print(iris[:,:4].sort())
iris[:,4]=iris[:,4].astype('str')
print(iris)
irispd=pd.DataFrame(iris)
print(irispd)
# column names of the dataframe
irispd.columns=names
print(irispd)

# convert the irispd dataframe to a numpy array
type(irispd.values)
irispdnum=irispd.values
irispdnuml=irispdnum.tolist()
print(irispdnuml)
print(iris.T)
print(iris)
# irispd
# irispdnuml to a numpy array
# group irispd by species
# irispd.groupby(['species','sepallength']).sum()
# irispd.columns
iris[:,:1]+iris[:,:2]
irispd[:3]
iris[:3]
iris[:,0]
print(irispd)
# getting the first column of irispd in numeric index
irispd.iloc[:,:4]
irispd.iloc[:,2:4].corr()
irispd.describe()
#sort a numpy array based on 2nd column
# sort descending the iris dataset based on sepallength column
iris[iris[:,0].argsort()][::-1][:20]
# iris[iris[:,0].argsort()][:20]
# iris[:,4].argsort()
# iris[:,:4]=iris[:,:4].astype('float')

# find the mean, median, standard deviation of iris's sepallength (1st column)
sepallength=iris[:,0]
print(sepallength)
print(np.mean(sepallength))
print(sepallength.mean())
print(np.median(sepallength))
print(np.std(sepallength))
# how to normalize an array so the values range exactly between 0 and 1
# normalize the values of sepallength
sepallength=iris[:,0]
print(sepallength)
smax=sepallength.max()
smin=sepallength.min()
print(smax)
print(smin)
s=(sepallength-smin)/(smax-smin)
s.size
# add s to the iris dataframe
iriss=np.vstack((iris.T,s))
irisnc=iris
# type(smax)
# insert the new column in iris
irisnc=np.insert(iris,0,s,axis=1)
print(irisnc)
print(iris)
# how to create a new column from existing columns of a numpy array
# create a new column for volume in iris dataset, where volume is (pi x petallength x sepallength^2)/3
sepallength=iris[:,0]
petallength=iris[:,2]
volume=(np.pi*petallength*(sepallength**2))/3
volume=volume[:,np.newaxis]
# add the new column to the iris dataset
iris=np.hstack([iris,volume])
print(iris)
# shuffle the columns of iris
np.random.shuffle(iris)
print(iris)
# interchange columns 1 and 2 in the array iris --important
iris[:,[0,1,2,3,4]]=iris[:,[4,0,1,2,3]]
print(iris)
# take a random sample of 20% of the rows of iris dataset
np.random.seed(100)
iristest=iris[np.random.randint(150,size=30),:] # testset important
iristest.size
# iris.size
# drop a column from a numpy array iris
iris[:,[0,1,2,3]]=iris[:,[1,2,3,4]] # important

# percentile scores of a numpy array
# compute the 5th and 95th percentile of sepallength
sepallength=iris[:,0]
sepallength
np.percentile(sepallength,q=[5,95])
iris[:,0:3].max()
# insert values at random positions in an array
# insert np.nan values at 20 random positions in iris dataset
iris[np.random.randint(150,size=20),np.random.randint(5,size=20)]=np.nan
print(iris)
# find the position of missing values in iris
# np.isnan(iris[:,4])
print(iris)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# Solution
print("Number of missing values: \n", np.isnan(iris_2d[:, 0]).sum())
print("Position of missing values: \n", np.where(np.isnan(iris_2d[:, 0])))

# filter a numpy array based on two or more conditions
# filter the rows of iris dataset that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0   #important
sepallength=iris[:,0]
petallength=iris[:,2]
# iris[(sepallength<5)&(petallength>1.5)]
iris[(iris[:,0]<5)&(iris[:,2]>1.5)]

# how to filter a numpy array based on two or more conditions
# filter the rows of iris dataset that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0   #important
sepallength=iris[:,0]
petallength=iris[:,2]
iris[(sepallength<5)&(petallength>1.5)]
# iris[(sepallength<5)//(petallength>1.5)].shape

# how to drop rows that contain a missing value from a numpy array
# drop rows that contain a missing value
iris_2d[np.sum(np.isnan(iris),axis=1)==0][:5]

# visualize the iris dataset and plot a histogram of the petallength
# visualize the petallength of the iris dataset
petallength=iris[:,2]
plt.hist(petallength)
# plt.show()
# bar plot of the petallength of the iris dataset
plt.bar(np.arange(150),petallength)

# how to find the correlation between two columns of a numpy array
# find the correlation between sepallength (1st column) and petallength (3rd column) in iris dataset
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_3 = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3,4])

# Solution 1
np.corrcoef(iris_3[:, 0], iris_3[:, 2])[0, 1]
np.corrcoef(iris_3[:, 0], iris_3[:, 2])
# iris_3

# How to do probabilistic sampling in numpy --important (need to revisit)
# Q. Randomly sample iris's species such that setose is twice the number of versicolor and virginica
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_4 = np.genfromtxt(url, delimiter=',', dtype='object')

# Approach 1: Generate Probablistically
np.random.seed(100)
a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])
np.random.choice(a, 150, p=[0.5, 0.25, 0.25])

# Approach 2: Probablistic Sampling (preferred) # created an index
np.random.seed(100)
probs = np.r_[np.linspace(0, 0.500, num=50), np.linspace(0.501, .750, num=50), np.linspace(.751, 1.0, num=50)]
index = np.searchsorted(probs, np.random.random(150))
species_out = species[index]
print(np.unique(species_out, return_counts=True))

species_out.shape
species_out
# how to get the second largest value of an array when grouped by another array
# what is the value of second longest petallength of species setosa
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_5 = np.genfromtxt(url, delimiter=',', dtype='object')
structured_array = np.core.records.fromarrays(iris_5.T, names=','.join(names)) # structured array in case operations need to be peformed
petallength=iris_5[:,2]
# Get the second last value
np.unique(np.sort(petallength))[-2]
print(iris_5)
# sort a 2d numpy array by a column
# sort the iris dataset based on sepallength column
iris_5[iris_5[:,0].argsort()][:20]

# unique values and counts of a column
np.unique(iris_5[:,3]).size

# find the most frequent value in a numpy array
# find the most frequent value of petal length (3rd column) in iris dataset
vals, counts = np.unique(iris[:, 1], return_counts=True)
print(vals[np.argmax(counts)],vals)

# How to find the position of the first occurrence of a value greater than a given value
# find the position of the first occurrence of a value greater than 1.0 in petalwidth 4th column of iris dataset
np.argwhere(iris[:,3].astype(float)>1.0)[0]

# how to replace all values greater than a given value to a given cutoff
# replace all values greater than 30 to 30 and less than 10 to 10 in a given numpy array
np.random.seed(100)
a=np.random.uniform(1,50,20)
print(a)
a[a>30]=30
a[a<10]=10
print(a)

# how to get the positions of top n values from a numpy array
# get the positions of top 5 maximum values in a given array a
np.random.seed(100)
a=np.random.uniform(1,50,20)
print(a)
a.argsort()
a[a.argsort()][-5:]

# how to compute the row wise counts of all possible values in an array
# compute the counts of unique values row wise
np.random.seed(100)
arr=np.random.randint(1,11,size=(6,10))
print(arr)
# need to work on this

# how to convert an array of arrays into a flat 1d array
# convert array_of_arrays into a flat linear 1d array --important --doesnt work
arr=np.arange(3)
print(arr)
arr1=np.arange(3,7)
print(arr1)
arr2=np.arange(7,10)
print(arr2)
array_of_arrays=np.array([arr,arr1,arr2])
print(array_of_arrays)
# arr_2d=np.concatenate(array_of_arrays)
# print(arr_2d)

# how to generate one-hot encodings for an array in numpy
# compute the one-hot encodings (dummy binary variables for each unique value in the array)
np.random.seed(101)
arr=np.random.randint(1,4,size=6)
print(arr)

# How to create row numbers grouped by a categorical variable
# create row numbers grouped by a categorical variable. Use the following sample from iris species as input
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_6 = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
iris_6
# need to complete this

# how to rank items in an array using numpy
# create the ranks for the given numeric array a
np.random.seed(10)
a=np.random.randint(20,size=10)
print(a)
a.argsort().argsort() #--important

# how to rank items in a multidimensional array using numpy
# create a rank array of the same shape as a given numeric array a
np.random.seed(10)
a=np.random.randint(20,size=[2,5])
print(a.ravel())
a.ravel().argsort().argsort().reshape(2,5)

# how to find the maximum value in each row of a numpy array 2d #important
# compute the maximum for each row in the given array
np.random.seed(100)
a=np.random.randint(1,10,[5,3])
print(a)
np.amax(a,axis=1) #important
# sol 2
np.apply_along_axis(np.max,arr=a,axis=1)
# sol 3
np.apply_along_axis(lambda x:np.max(x),arr=a,axis=1)

#how to compute the min-by-max for each row for a numpy array 2d #very important since it involves lambda
# compute the min-by-max for each row for given 2d numpy array
np.random.seed(100)
a=np.random.randint(1,10,[5,3])
print(a)
print(np.apply_along_axis(lambda x:np.min(x)/np.max(x),arr=a,axis=1))

# how to find the duplicate records in a numpy array #important
# find the duplicate entries (2nd occurrence onwards) in the given numpy array and mark them as True. First time occurrences should be False
np.random.seed(100)
a=np.random.randint(0,5,10)
print(a)
out=np.full(a.shape[0],True)
print(out)
#find the index positions of unique elements
unique_positions=np.unique(a,return_index=True)[1]
print(unique_positions)
#mark those positions as False
out[unique_positions]=False
print(out)
# how to find the grouped mean in numpy
# find the mean of a numeric column grouped by a categorical column in a 2d numpy array
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_7 = np.genfromtxt(url, delimiter=',', dtype='object')
print(iris_7)
names=('sepallength','sepalwidth','petallength','petalwidth','species')

names=['sepallength','sepalwidth','petallength','petalwidth','species']


# easy implementation in pandas


# how to convert a PIL image to numpy array
# Import modules
from io import BytesIO
from PIL import Image
import numpy as np
URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'


# how to drop all missing values from a numpy array #important
# drop all nan values from a 1d numpy array
a=np.array([1,2,3,np.nan,5,6,7,np.nan])
a
a[~np.isnan(a)]
# how to compute the euclidean distance between two arrays
# compute the euclidean distance between two arrays a and b
a=np.array([1,2,3,4,5])
b=np.array([4,5,6,7,8])
dist=np.linalg.norm(a-b)
print(dist)
# how to find all the local maxima (or peaks) in a 1d array
# find all the peaks in a 1d numpy array a. Peaks are points surrounded by smaller values on both sides
a=np.array([1,3,7,1,2,6,0,1])
# sol 1
doublediff=np.diff(np.sign(np.diff(a)))
print(doublediff)

# how to compute the moving average of a numpy array #important or do the python list implemantation
# compute the moving average of window size 3, for the given 1d array
np.random.seed(100)
Z=np.random.randint(10,size=10)
print(Z)
# sol 1
np.convolve(Z,np.ones(3)/3,mode='valid')
# sol 2
np.cumsum(Z)
np.cumsum(Z)[3:]
np.cumsum(Z)[3:]/3

# how to create a numpy array sequence given only the starting point, length and the step
# create a numpy array of length 10, starting from 5 and has a step of 3 between consecutive numbers
np.arange(5,5+10*3,3)

# how to fill in missing dates in an irregular series of numpy dates
# given an array of a non-continuous sequence of dates. Make it a continuous sequence of dates, by filling in the missing dates
dates=np.arange(np.datetime64('2018-02-01'),np.datetime64('2018-02-25'),2)
print(dates)

# how to create strides from a given 1d array
# from the given 1d array arr, generate a 2d matrix using strides, with a window length of 4 and strides of 2, like [[0,1,2,3],[2,3,4,5],[4,5,6,7]...]
# need to work on this

block_1 = np.array([[1, 1], [1, 1]])
block_2 = np.array([[2, 2, 2], [2, 2, 2]])
block_3 = np.array([[3, 3], [3, 3], [3, 3]])
block_4 = np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]])

block_new = np.block([
    [block_1, block_2],
    [block_3, block_4]
])

print(block_new)


array_1 = np.array([[1, 2], [3, 4]])
array_2 = np.array([[5, 6], [7, 8]])

array_new = np.concatenate((array_1, array_2), axis=1)
print(array_new)


array_1 = np.array([1, 2, 3, 4])
array_2 = np.array([5, 6, 7, 8])
array_new = np.stack((array_1, array_2), axis=1)
print(array_new)
# Array initialization

full_array_2d = np.full((3, 4), 5)
print(full_array_2d)

# Create an empty array
empa = np.empty((3, 4), dtype=int)
print("Empty Array")
print(empa)

# Create a full array
flla = np.full([3, 3], 55, dtype=int)
print("\n Full Array")
print(flla)



# numpy slicing
# 1. create a numpy array
a=np.arange(15)
print(a)
# 2. slice the numpy array
print('a[3:8]',a[3:8])

# 3. slice the numpy array
print('a[3:8:2]',a[3:8:2])
# 4. slice the numpy array
print('a[::-1]',a[::-1])
# 5. slice the numpy array
print('a[5:]',a[5:])
# 6. slice the numpy array
print('a[:5]',a[:5])
# 7. slice the numpy array
print('a[::2]',a[::2])
# 8. slice the numpy array
print('a[1::2]',a[1::2])
# 9. slice the numpy array
print('a[1:5:2]',a[1:5:2])
# 10. slice the numpy array
print('a[1:5:3]',a[1:5:3])
# 11. slice the numpy array
print('a[1:10:5]',(a[1:5:4]))
# 12. slice the numpy array
# Common compare and contrast across data structures
# slicing/index,conversion,methods,functions,visualization,importing/exporting,missing values,groupby,merge,join,concatenate,reshape,iteration,sorting,filtering,aggregation,scaling,normaliz
# numpy 2d array initialization
# 1. create a numpy 2d array
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
# numpy 3d array initialization
# 1. create a numpy 3d array
a3=np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])
# numpy array with strings
# 1. create a numpy array with strings
# a=np.array(['a','b','c'])
# numpy array with strings and numbers
# 1. create a numpy array with strings and numbers
am=np.array([1,2,'a','b'])

# numpy 2d slicing


# Create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Slice the first row
print(arr[0,:])

# Slice the second column
print(arr[:,1])

# Slice the last two rows
print(arr[1:3,:])

# Slice the last two columns of the first row
print(arr[0,-2:])

# Slice the elements from index 1 to 4 from the second row
print(arr[1,1:4])

################ Numpy end    #########
################ stats     ############
# In[11]:

df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]})

print(df.describe())

df = pd.read_csv('ds_salaries.csv', index_col=0)

df.iloc[df['sales'].idxmax()][['gender', 'income']]

df.sample(3)
df.shape
df.isnull().sum().sum()
df.size
df.dtypes
df['gender'].unique()
df.describe()
df['income'].nunique()
df['income'].is_monotonic_increasing
df['income'].value_counts(normalize=True)
df['income'].nsmallest(3)
df['income'].nlargest(3)
df.corr()
df.plot(x='income', y='sales', kind='scatter')
df['income'].plot(kind='hist')

df.dropna()
df.dropna(axis=1)

# Subsets - rows and columns
# Use df.loc[] and df.iloc[] to select only
# rows, only columns or both.
# Use df.at[] and df.iat[] to access a single
# value by row and column.
# First index selects rows, second index columns.


# Create DataFrame with a MultiIndex


# Reshaping Data – Change layout, sorting, reindexing, renaming

df = pd.DataFrame(
{"a" : [4, 5, 6],
"b" : [7, 8, 9],
"c" : [10, 11, 12]},
index = [1, 2, 3])

# Specify values for each column.
df = pd.DataFrame(
[[4, 7, 10],
[5, 8, 11],
[6, 9, 12]],
index=[1, 2, 3],
columns=['a', 'b', 'c'])
#Specify values for each row.


df[df.a > 7]


df.drop_duplicates()

df.b
df.iloc[10:20]
df.sample(frac=0.5)
df.filter(regex='regex')
df.iloc[:, [1, 2, 5]]
df.sample(n=10) # Randomly select n rows.
df.nlargest(3, a)
df.loc[:, 'x2':'x4']
df.nsmallest(3, 'value')
df.loc[df['a'] > 10, [a, c]]
df.head(5)
df.tail(5)
df.query('a > 7')
df.iat[1,2] # Access single value by index
df.query('Length > 7 and Width < 8')
# df.query('Name.str.startswith("abc")', df.at[4, 'a'] # Access single value by label
# df.column.isin(values)

# Summarize Data
df['w'].value_counts()
len(df) # of rows in DataFrame.
df.shape
df['w'].nunique() # of distinct values in a column.
df.describe()
df.sum()
df['w'].min()
df.count()
df.max()
df.mean()
df.median()
df.var()
df.std()
first = ['F1', 'F1', 'F1', 'F2', 'F2', 'F2']
second = [1, 2, 3, 1, 2, 3]
top_index = list(zip(first, second))
top_index = pd.MultiIndex.from_tuples(top_index)

df = pd.DataFrame(np.random.randn(6, 2), top_index, ['A', 'B'])


df.fillna(value=df['A'].mean())
'''
df = pd.read_csv('data.csv')
df = df.fillna(...)
df = df.query('some_condition')
df['new_column'] = df.cut(...)
df = df.pivot_table(...)
df = df.rename(...)


df = (pd.read_csv('data.csv')
      .fillna(...)
      .query('some_condition')
      .assign(new_column=df.cut(...))
      .pivot_table(...)
        .rename(...)
)
'''

# pandascheatsheet

# http://pandas.pydata.org


df.sort_values('mpg')# Order rows by values of a column (low to high).
# df.sort_values('mpg’, ascending=False) # Order rows by values of a column (high to low).
pd.melt(df)#Gather columns into rows.

df.pivot(columns='var', values='val')# Spread rows into columns.


df = (pd.melt(df) # important
.rename(columns={
'variable':'var',
'value':'val'})
.query('val >= 200')
)

df.rename(columns = {'y':'year'}) # Rename the columns of a DataFrame
df.sort_index() # Sort the index of a DataFrame
df.reset_index() # Reset index of DataFrame to row numbers, moving
# index to columns.

pd.concat([df,df]) # Append rows of DataFrames

# Subset Observations - rows

pd.concat([df,df], axis=1) # Append columns of DataFrames

# Subset Variables - columns

# df.drop(columns=['Length’, 'Height']) # Drop columns from DataFrame


################ stats end ############

################ pandas     ############
# In[11]:

data = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names) # data = load_iris()

data.rename(columns={'sepal length (cm)': 'length', 'sepal width (cm)': 'width', 'petal length (cm)': 'plength', 'petal width (cm)': 'pwidth'}, inplace=True)

print(data.head())
df1 = pd.read_csv('cars.csv')
df1.columns = ['Height', 'Length', 'Width', 'Driveline', 'Engine Type', 'Hybrid', 'Number of Forward Gears',
               'Transmission', 'City-mpg', 'Fuel-Type', 'Highway-mpg', 'Classification', 'ID', 'Make', 'Model-Year',
               'Year', 'HorsePower', 'Torque']

df1.to_csv('cars.csv')

pysqldf = lambda q: sqldf(q, globals())  # this should suffice

q="""
    with T as 
    (
    select * from data
    where length>4
    ),
    T1 as
    (
    select * from T
    where length>5.3
    )
    select * from T1
  """
qc = """
CREATE TABLE nt (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER
)
"""
qc1 = """
INSERT INTO nt (id, name)
VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Carol');
"""
dfc=pysqldf(qc)
print(dfc)

dfq1 = pysqldf(q)
print(dfq1)

# Lamda functions very important
CB = 0
def define_test(X1, X2):  # ,X2,X3):
    CB=0
    if CB == 0:
        return X1 * X2

    elif X1 == CB:
        return X2
    #         elif X2 !=0:
    #         return X2
    #     elif X3 !=0:
    #         return X3
    else:
        return X1


# ClusDFOverall['ClusT'] = ClusDFOverall.apply(lambda x: define_cluster(x['ClusT'],x['Clus14']),axis=1)
# df1['Area'] = df1.apply(lambda x: define_test(x['Height'], x['Length']), axis=1)  # very important

data['Area'] = data.apply(lambda x: define_test(x['length'], x['width']), axis=1)  # very important


# initialize a dataframe
df = pd.DataFrame(
    [[21, 72, 67],
     [23, 78, 69],
     [32, 74, 56],
     [52, 54, 76]],
    columns=['a', 'b', 'c'])

# pandas slicing

# Create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Slice the DataFrame by index
df[1:3]

# Slice the DataFrame by index and column
df[1:3, 'A']

# Slice the DataFrame by index and column using loc
df.loc[1:3, 'A']

# Slice the DataFrame by index and column using iloc
df.iloc[1:3, 0]

# Create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# Slice the DataFrame by index using loc
df.loc[1:3, 'A']

# Create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Slice the DataFrame by index using iloc
df.iloc[1:3, 0]

# Create a DataFrame
df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Carol', 'Dave'],
                   'Age': [20, 25, 30, 35]})

# Select all rows where Age is greater than 25
df.loc[df['Age'] > 25]

# Select the row for 'Carol'
df.loc[df['Name'] == 'Carol']

# Select the first two rows (conceptually same for lists and numpy too)
df.loc[:2]

# Select the last two rows
df.loc[-2:]

# Select the first two columns
df.loc[:, :2]

# Select the last two columns
df.loc[:, -2:]

# Select the 'Name' and 'Age' columns
df.loc[:, ['Name', 'Age']]

# Select all rows where Age is greater than 25 and Name is 'Carol'
df.loc[(df['Age'] > 25) & (df['Name'] == 'Carol')]

# Create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]})

# Slice the DataFrame by row index
df.iloc[0:3]

# Slice the DataFrame by column index
df.iloc[:, 0:2]

# Slice the DataFrame by row and column index
df.iloc[0:3, 0:2]

# Slice the DataFrame by boolean mask
df[df['A'] > 3]


# Slice the DataFrame by callable function
def g(x):
    return x > 3


df[g(df['A'])]

# pandas groupby

# Create a DataFrame
df = pd.DataFrame({'Name': ['John', 'Mary', 'John', 'Mary'],
                   'Age': [20, 25, 22, 28]})

# Group the data by the 'Name' column
grouped = df.groupby('Name')

# Calculate the mean and standard deviation of the 'Age' column for each group
agg_results = grouped['Age'].agg(['mean', 'std'])

# Print the results
print(agg_results)

# pandas pivot

# Create a DataFrame
df = pd.DataFrame({'Name': ['John', 'Mary', 'John', 'Mary'],
                   'Age': [20, 25, 22, 28],
                   'City': ['London', 'New York', 'London', 'Paris']})

# Pivot the DataFrame
df = df.pivot(index='Name', columns='City', values='Age')

# Print the DataFrame
print(df)
# pandas melt
# Create a DataFrame in wide format
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
# Unpivot the DataFrame
df_long = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
# Print the long format DataFrame
print(df_long)

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# Select a column
df['A']

# Select multiple columns
df[['A', 'B']]

# pandas filtering
# Select rows where the value in column 'A' is greater than 2
df[df['A'] > 2]

# Select rows where the value in column 'A' is equal to 2 and the value in column 'B' is equal to 5
df[(df['A'] == 2) & (df['B'] == 5)]

# Sort the DataFrame by the value in column 'A'
df.sort_values('A')

# Group the DataFrame by the value in column 'A' and calculate the mean of the values in column 'B'
df.groupby('A')['B'].mean()

different_aggregations = df.groupby(['Role', 'Gender']).agg({
    'Years_Experience': 'max',
    'Salary': ['mean', 'median']
})
print(different_aggregations)

# Create a new column called 'C' that is the sum of the values in columns 'A' and 'B'
df['C'] = df['A'] + df['B']

# Drop the column 'C'
df.drop('C', axis=1)

# pandas index and reindex

df.index

df.reindex('A', fill_value=0)

# import numpy and pandas module

column = ['a', 'b', 'c', 'd', 'e']

index = ['A', 'B', 'C', 'D', 'E']

# create a dataframe of random values of array

df1 = pd.DataFrame(np.random.rand(5, 5),columns=column, index=index)

print(df1)

print('\n\nDataframe after reindexing rows: \n',df1.reindex(['B', 'D', 'A', 'C', 'E']))

column = ['a', 'b', 'c', 'd', 'e']

index = ['A', 'B', 'C', 'D', 'E']

# create a dataframe of random values of array

df1 = pd.DataFrame(np.random.rand(5, 5),columns=column, index=index)

# create the new index for rows

new_index = ['U', 'A', 'B', 'C', 'Z']

print(df1.reindex(new_index))
# exercise -- aggregate (if possible also try melt and join the aggregated table to the main table
url1='https://raw.githubusercontent.com/Sketchjar/MachineLearningHD/main/iris.csv'
ir1=np.genfromtxt(url1,delimiter=',',dtype=None)

iris_7pd=pd.DataFrame(ir1)
names=['SN','sepallength','sepalwidth','petallength','petalwidth','species']

iris_7pd = iris_7pd.iloc[1:] # droping the first row of the dataframe
iris_7pd.columns = names
iris_7pd[['SN','sepallength','sepalwidth','petallength','petalwidth']]=iris_7pd[['SN','sepallength','sepalwidth','petallength','petalwidth']].astype(float)

iris_7pd['species']=iris_7pd['species'].astype('string')
x='species'


irisgrouped=iris_7pd.groupby(x).mean().reset_index()

# Merge on the 'key' column
result = pd.merge(iris_7pd, irisgrouped, on='species')


# exercise end

# try parametrizing everything
################ pandas end ############

########## Most consumate concise printout worthy ready reckoner for Regex, Strings, Date, Tips, pandas ...(stats) end ######
# In[11]:
# classes & Exception handling and logging
# Base class
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound."

# Derived class
class Dog(Animal):
    def __init__(self, name='buddy', breed='GR'):
        # Call the constructor of the base class
        super().__init__(name)
        self.breed = breed

    def speak(self):
        # Override the speak method
        return f"{self.name}, the {self.breed}, barks."

# Usage
dog1 = Dog()
print(dog1.speak())

dog = Dog("Buddy", "Golden Retriever")
print(dog.speak())

import logging
class CustomException(Exception):
    def __init__(self, message):
        super().__init__(message)
n=10
# Example usage
try:
    if n > 5:
        # raise CustomException("This is a custom exception!")
        raise CustomException("Exception")#("This is a custom exception!")
    res=10/0
except CustomException as e:
    print(f"Caught an exception: {e}")# Configure logging
logging.basicConfig(
    filename='error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    # Code that may raise an exception
    result = 10 / 0
except Exception as e:
    logging.error("An exception occurred", exc_info=True)

# Regular DS and Operations

# python operators
# Example
a = 10
b = 5

# Addition
print(a + b)  # Output: 15

# Subtraction
print(a - b)  # Output: 5

# Multiplication
print(a * b)  # Output: 50

# Division
print(a / b)  # Output: 2.0

# Modulus
print(a % b)  # Output: 0

# Exponentiation
print(a ** b)  # Output: 100000

# Floor division
print(a // b)  # Output: 2


# Bitwise AND operator
a = 60  # 111100
b = 13  # 001101
c = a & b  # 001100
print(c)  # 12

# Bitwise OR operator
a = 60  # 111100
b = 13  # 001101
c = a | b  # 111101
print(c)  # 61

# Bitwise XOR operator
a = 60  # 111100
b = 13  # 001101
c = a ^ b  # 110001
print(c)  # 49

# Bitwise NOT operator
a = 60  # 111100
b = ~a  # 000011
print(b)  # 11

# Left shift operator
a = 60  # 111100
b = a << 2  # 11110000
print(b)  # 240

# Right shift operator
a = 60  # 111100
b = a >> 2  # 001111
print(b)  # 15

## Dict Methods--Copy,pop,update,clear

x=[1,2,3,4]
y=['a','b','c','d']
z=dict.fromkeys(x, y)

print(z)

color = {"c1": "Red", "c2": "Green", "c3": "Orange"}
co=color.copy()
print(co)

print(color.items())
print(color.popitem())
print(color.keys())
print(list(color.values()))

print(color.update({"c4": "White"}))

## Dict Sorting

orders = {
	'cappuccino': 54,
	'latte': 56,
	'espresso': 72,
	'americano': 48,
	'cortado': 41
}

sort_orders = sorted(orders.items(), key=lambda x: x[1])

print(color.popitem())

#List Method
# List Methods
# append()	Adds an element at the end of the list
# clear()	Removes all the elements from the list
# copy()	Returns a copy of the list
# count()	Returns the number of elements with the specified value
# extend()	Add the elements of a list (or any iterable), to the end of the current list
# index()	Returns the index of the first element with the specified value
# insert()	Adds an element at the specified position
# pop()	Removes the element at the specified position
# remove()	Removes the first item with the specified value
# reverse()	Reverses the order of the list
# sort()	Sorts the list

# sets are define
A = {0, 2, 4, 6, 8};
B = {1, 2, 3, 4, 5, 4}

# union
print("Union :", A | B)
# intersection
print("Intersection :", A & B)
# difference
print("Difference :", A - B)
# symmetric difference
print("Symmetric difference :", A ^ B)
print((A - B)|(B-A))
print((A & B))

C=str(B)
print(len(B))

#Train test split
from sklearn.model_selection import train_test_split

# Example DataFrame
Chicken_SWC = {
    'X': [1, 2, 3, 4, 5],
    'SALESQTY': [3, 4, 12, 15, 10],
    'Y': [10, 20, 15, 25, 30],
    'Z': [13, 40, 15, 12, 10]
}
Chicken_SWC = pd.DataFrame(data)

train, test = train_test_split(Chicken_SWC, test_size = 0.2)
# In[29]:
cols = Chicken_SWC.columns.tolist()
feature_cols=cols
# len(feature_cols)
# feature_cols=feature_cols[:1]+feature_cols[2:]
feature_cols=feature_cols[:3]

X=train[feature_cols]
y=train.SALESQTY
X.head()

##################### Visualization #########################

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
y1 = [150, 20, 250, 500]

# Example DataFrame
data = {
    'X': [1, 2, 3, 4, 5],
    'Y': [10, 20, 15, 25, 30],
    'Z': [13, 40, 15, 12, 10],
    'SALESQTY': [3, 4, 12, 15, 10]
}
df = pd.DataFrame(data)
plt.bar(df['X'], df['Y'], facecolor='none', edgecolor='b',zorder=2) #main df
plt.show()

plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# plt.grid(True)
# Plotting the data
plt.bar(x, y, facecolor='none', edgecolor='b',zorder=2) #main
plt.bar(df['X'], df['Y'], facecolor='none', edgecolor='b',zorder=2) #main df
# plt.boxplot(x, y,zorder=2)
# plt.hist(x, y,zorder=2)
plt.scatter(x, y,zorder=2)

# Create a secondary y-axis
ax1 = plt.gca()  # Get the current axis
ax2 = ax1.twinx()  # Create a twin axis sharing the same x-axis
ax2.bar(x, y1, facecolor='none', edgecolor='r', label='Secondary Axis',zorder=0)
ax2.legend()
# Display the plot
plt.show()
##################### Visualization end ######################
##################### Tensors #########################
import torch
tx=torch.from_numpy(arr)
print(tx.dtype)
print(tx.type)

arr2 = np.arange(0.,12.).reshape(4,3)
tx2 = torch.from_numpy(arr2)
tx3=torch.tensor(arr2)
tx4=torch.empty(4,3, dtype=torch.int64)
tr=torch.rand(4,3)
tri=torch.randint(0, 50, (4, 3))
torch.rand_like(tr)
torch.ones_like(tr)

# understanding manual seed
np.linspace(2,10,8, dtype=np.int64).reshape(2,4)
tr.shape
tr.size()

# view and reshape
tr.view(-1,2) # -1 for the infered dimension
# operations
torch.add(tr,tr)
torch.add(tr,tr).sum()
tr@torch.rand(3,4)
torch.abs(tr)
(tr@torch.rand(3,4)).size()
(tr@torch.rand(3,4)).numel() # number of elements

##################### Tensors end #########################





























































































