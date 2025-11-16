
# coding: utf-8

# https://www.w3schools.com/python/python_regex.asp

# In[2]:


# Searching for a match

import re

txt = "The rain in Spain"
x = re.search("^The.*Spain$", txt)
if (x):
  print("YES! We have a match!")
else:
  print("No match")


# In[7]:


import re

str = "The rain in Spain"
x = re.findall("ai", str)
print(x)


# In[9]:


# Search for empty string
import re

str = "The rain in Spain"
x = re.search("\s", str)

print("The first white-space character is located in position:", x.start())


# In[11]:


# search that returns no match
str = "The rain in Spain"
x = re.search("Portugal", str)
print(x)


# In[13]:


# Split at each white-space character:
str = "The rain in Spain"
x = re.split("\s", str)
print(x)


# In[4]:


# Replace every white-space character with the number 9:
str = "The rain in Spain"
x = re.sub("\s", "9", str)
print(x)
# Replace the first 2 occurrences
str = "The rain in Spain"
x = re.sub("\s", "9", str,2)
print(x)


# In[16]:


# Do a search that will return a Match Object:
str = "The rain in Spain"
x = re.search("ai", str)
print(x) #this will print an object


# In[19]:


# Print the position (start- and end-position) of the first match occurrence
# The regular expression looks for any words that starts with an upper case "S":

str = "The rain in Spain"
x = re.search(r"\bS\w+", str)
print(x.span())
print(x.group())


# In[18]:


# Print the part of the string where there was a match.
# The regular expression looks for any words that starts with an upper case "S":
str = "The rain in Spain"
x = re.search(r"\bS\w+", str)
print(x.group())


# Metacharacters
# 
# ### Some return characters and some return words

# In[28]:


str = "The rain in Spain"

################# Find all lower case characters alphabetically between "a" and "m": ################# 

x = re.findall("[a-m]", str)
print(x)


#################  Signals a special sequence (can also be used to escape special characters) ################# 
str = "That will be 59 dollars"

#Find all digit characters:
x = re.findall("\d", str)
print(x)


################# Any character (except newline character) ################# 
str = "hello world"
#Search for a sequence that starts with "he", followed by two (any) characters, and an "o":
x = re.findall("he..o", str)
print(x)


################# Starts with ################# 
str = "hello world"

#Check if the string starts with 'hello':

x = re.findall("^hello", str)
if (x):
  print("Yes, the string starts with 'hello'")
else:
  print("No match")

################# Ends with ################# 
str = "hello 23 world"

#Check if the string ends with 'world':

x = re.findall("world$", str)
if (x):
  print("Yes, the string ends with 'world'")
else:
  print("No match")

x = re.findall("[a-m]", str)
print(x)

#Find all digit characters:

x = re.findall("\d", str)
print(x)


# In[29]:


#Check if the string ends with 'world':

x = re.findall("world$", str)
if (x):
  print("Yes, the string ends with 'world'")
else:
  print("No match")

print(x)


# In[33]:


# Zero or more occurrences

str = "The rain in Spain falls mainly in the plain!"

#Check if the string contains "ai" followed by 0 or more "x" characters:

x = re.findall("ain*", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[36]:


# One or more occurrences
str = "The rain in Spain falls mainly in the plain!"

#Check if the string contains "ai" followed by 1 or more "x" characters:

x = re.findall("ainl+", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[39]:


# Excactly the specified number of occurrences
str = "The rain in Spain falls mainly in the plain!"

#Check if the string contains "a" followed by exactly two "l" characters:

x = re.findall("al{1}", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[40]:


# Either or
str = "The rain in Spain falls mainly in the plain!"

#Check if the string contains either "falls" or "stays":

x = re.findall("falls|stays", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[50]:


# \A Returns a match if the specified characters are at the beginning of the string

str = "The rain in Spain"

#Check if the string starts with "The":

x = re.findall("\AThe", str)

print(x)

if (x):
  print("Yes, there is a match!")
else:
  print("No match")

#\b Returns a match where the specified characters are at the beginning or at the end of a word
str = "The rain in Spain"

#Check if "ain" is present at the beginning of a WORD:
x = re.findall(r"\bThe", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

# Returns a match where the string contains digits (numbers from 0-9)
str = "The rain in Spain"

#Check if the string contains any digits (numbers from 0-9):

x = re.findall("\d", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

# \D Returns a match where the string DOES NOT contain digits
str = "The rain in Spain"

#Return a match at every no-digit character:

x = re.findall("\D", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

#\s Returns a match where the string contains a white space character
str = "The rain in Spain"

#Return a match at every white-space character:

x = re.findall("\s", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

#\S Returns a match where the string DOES NOT contain a white space character
str = "The rain in Spain"

#Return a match at every NON white-space character:

x = re.findall("\S", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

#  \w Returns a match where the string contains any word characters 
# (characters from a to Z, digits from 0-9, and the underscore _ character)
str = "The rain in Spain"

#Return a match at every word character (characters from a to Z, digits from 0-9, and the underscore _ character):

x = re.findall("\w", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

# \W Returns a match where the string DOES NOT contain any word characters
str = "The rain in Spain"

#Return a match at every NON word character (characters NOT between a and Z. Like "!", "?" white-space etc.):

x = re.findall("\W", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

#\Z Returns a match if the specified characters are at the end of the string
str = "The rain in Spain"

#Check if the string ends with "Spain":

x = re.findall("Spain\Z", str)

print(x)

if (x):
  print("Yes, there is a match!")
else:
  print("No match")


# Sets
# A set is a set of characters inside a pair of square brackets [] with a special meaning:

# In[65]:


#############################################################
# [arn]	Returns a match where one of the specified characters (a, r, or n) are present
str = "The rain in Spain"

#Check if the string has any a, r, or n characters:

x = re.findall("[arn]", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

# [a-n]	Returns a match for any lower case character, alphabetically between a and n

#############################################################
str = "The rain in Spain"

#Check if the string has any characters between a and n:

x = re.findall("[a-n]", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

#############################################################

# [^arn]	Returns a match for any character EXCEPT a, r, and n

str = "The rain in Spain"

#Check if the string has other characters than a, r, or n:

x = re.findall("[^arn]", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

#############################################################
# [0123]	Returns a match where any of the specified digits (0, 1, 2, or 3) are present

str = "8 times before 11:45 AM"

#Check if the string has any digits:

x = re.findall("[0-9]", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

#############################################################
# [0-5][0-9]	Returns a match for any two-digit numbers from 00 and 59

str = "8 times before 11:45 AM"

#Check if the string has any two-digit numbers, from 00 to 59:

x = re.findall("[0-5][0-9][0-9]", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

# [a-zA-Z]	Returns a match for any character alphabetically between a and z, lower case OR upper case
str = "8 times before 11:45 AM"

#Check if the string has any characters from a to z lower case, and A to Z upper case:

x = re.findall("[a-zA-Z]", str)

print(x)
print(type(x))

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

# [+]	In sets, +, *, ., |, (), $,{} has no special meaning, so [+] means: 
# return a match for any + character in the string

str = "8 times before +11:45 AM"

#Check if the string has any + characters:

x = re.findall("[+]", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[76]:



str = "433 times before 11:45 AM"
x = re.findall("[0-5][8-9][0-9]", str)

print(x)
print(type(x))
if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[73]:


lst = ['foo.py', 'bar.py', 'baz.py', 'qux.py', 'qux.py']


# In[75]:


s = set(lst)
s


# In[69]:


s = set()
for item in lst:
    s.add(item)


# In[71]:


list(s)


# In[72]:


set(s)


# In[ ]:


# list,tuple,dict


# In[ ]:


a = "Hello, World!"
print(a[1])


# In[ ]:


# Sub-String--Like list slicing
b = "Hello, World!"
print(b[2:5])


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


# In[ ]:


# Command-line String Input
print("Enter your name:")
x = input()
print("Hello, " + x)


# In[5]:


import datetime
#Display Weekday as a number 0-6, 0 is Sunday
i=2018
j=6
k=1
x = datetime.datetime(i, j, k)
#x=x+1
print(x.strftime("%w"))
y=x+datetime.timedelta(days=5)
x
type(y)
print(y.strftime("%B"))
print(y.strftime("%m"))
print(y.strftime("%j"))
print(y.strftime("%U"))
type(y.strftime("%B"))
z=int(y.strftime("%m"))+12


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
  
map(lambda x, y: x + y, list_a, list_b) # Output: [11, 22, 33]


# In[9]:


list(map(lambda x, y: x + y, list_a, list_b))


# ### Dict comprehension

# In[10]:


fruits = ['apple', 'mango', 'banana','cherry']
{f:len(f) for f in fruits}


# In[11]:


dict1 = {1: 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
# Double each value in the dictionary
double_dict1 = {k:v*2 for (k,v) in dict1.items() if v>3}
print(double_dict1)


# In[13]:


double_dict1['d']


# ### Recursion

# In[ ]:


def Rec(n):
    if n>1:
        return n*Rec(n-1)
    else:
        return 1
        
Rec(3)# Think in terms of Brackets or Tree  


# In[ ]:


T="()"
type(T)

LT=list(T)
LT[1]


# #### Args & Kwargs

# In[14]:

Dec=[1,2] # added to sortout  the issue
def Decoder(a):
    b=[]
    for i in range(len(a)):
        print(i)
        print(Dec[1])
        b.append(Dec[i])
        print(i)
    return b
D=Decoder(a)
D
type(D)


# In[21]:


def Decoder(*args):
    for i in args:
        for j in i:
            print(j)
        
#         print(i)

D=Decoder([1,2],[3,4])

