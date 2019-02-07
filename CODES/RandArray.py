####Task 1
from pip._vendor.distlib.compat import raw_input


def factorial(x):
    i = 0
    v_total = 1
    while i <= x:
        v_total *= x
        x = x-1
        i += 1
    return v_total

#Test Task 1
print (factorial(4));

####Task 2
def permutation(x,y):
    return factorial(x)/factorial(x-y)

print(permutation(3,4))

####Task 3
def combination(x,y):
    return permutation(x,y)/(factorial(y))

print(combination(4,5))

####Task 4
#Easy Way
import numpy as np
import random
randArray = np.random.randint(17,38, size = (6,8), dtype = int)
print(randArray)


#Hard Way
import numpy as np
import random
num_arr = np.zeros((6,8)) #np.zeros(shape=(6,8)) ile aynı
for i in range(0,6):
    for j in range(0,8):
        num_arr[i,j] = random.randint(17,38)
print(num_arr)


####Task 5
def is_prime(n):
    for i in range(2, int(n**0.5)+1):
        if n%i==0:
            return False
    return True

def is_prime_value(n):
    for i in range(2, int(n**0.5)+1):
        if n%i==0:
            return False
    return n

import numpy as np
i=2
ind=0
num_arr = []
while ind <100:
    if(is_prime_value(i) != False):
        num_arr.append(is_prime_value(i))
        ind += 1
        i+= 1
    else:
        i += 1
print(num_arr)

####Task 6

def first_primes(n):
    i = 2
    lst = []
    while (len(lst)<n):
        if is_prime(i):
            lst.append(i)
        i += 1
    return lst

print(first_primes(100))


####Task 7
def fibo(n):
    if n < 2:
        return n
    else:
        return fibo(n-1)+fibo(n-2)

print(fibo(8))


####Task 8
def rec_power(x,y):
    v_result = 1
    if y==0:
        return 1
    elif y==1:
        return x
    for i in range(1,y+1):
        v_result *=x
    return v_result

print(rec_power(3,1))


def f_power(x,y):
    if y == 0:
        return 1
    else:
        return x*f_power(x,y-1)

print(f_power(2,4))


####Task 9
def cumSumLoop(n,m):
    sum = 0
    for i in range(n, m+1):
        sum += i
    return sum

print(cumSumLoop(2,4))

def cumSumRec(n,m):
    if n > m:
        return 0
    return n + cumSumRec((n+1),m)

print(cumSumRec(2,4))
