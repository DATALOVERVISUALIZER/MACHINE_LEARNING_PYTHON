#1##Get two Boolean variable and get results in case of and, or
# alternatif 1
# XOR ,Exclusive OR iki string birbirinden farklıysa TRUE, aynı ise FALSE result verir.
def logical_xor(a, b):
    if bool(a) == bool(b):
        return False
    else:
        return a or b

# Alternatif 2
a = True
b = False
if (a and b):
    print("both are true")
elif (a):
    print("a true and b false ")
elif(b):
    print("b false and a true")
else:
    print("no of them true")

##alternatif 3
p1=True
p2=False
if (p1 and p2):
    print("false")
elif (p1 or p2):
    print("true") ## print yazılırken () prantez kullanılır.
else:
    print("false")

# 2# Procedures
def meanOfTwoNum(a, b):
    a = 2
    b = 6
    return ((a + b) / 2)

print (meanOfTwoNum(2,6))

# 3.#find the biggest ad bigger numbers
def bigger(a, b):
    if a > b:
        return a
    else:
        return b

print (bigger(14,15))

def  biggest(a,b,c):
    return bigger(bigger(a,b),c)

print (biggest(3,4,5))

# second way
def  biggest(a,b,c):
    if c>(bigger(a,b)):
        return c
    else:
        return bigger(a,b)

print(biggest(5,8,9))

a = True
b = False
if (a and b):
    print("both are true")
elif (a):
    print("a true and b false ")
elif (b):
    print("b false and a true")
else:
    print("no of them true")

                        ###################LOOP####################################
i = 1
sum=0
while i < 200: #ctrl+alt+l codu formatlamak için kullanılır.
    sum=sum+i
    i = i + 1
    print (sum)

# # ikinci yöntem
for i in range(1, 200,2): #2 incrementration için kullanılıyor.
    print(i)


# ##toplam fonsiyonunda ilk girilen sayı 5, ikincisi 15 5den 15'e kadar olan sayıların toplamını vereceğiz. Min ve max şeklinde sayıları
def func(minvar, maxvar):
    sum = 0
    while minvar <= maxvar:
        sum +=minvar
        minvar=minvar+1
    return sum
        # sum = sum + minvar
        # minvar=minvar+1


print (func(2,4))

# second way
def listsum(numList):
    theSum = 0
    for i in numList:
        theSum = theSum + i
    return theSum

print(listsum([1,3,5,7,9]))

# Girilen rakamın factorial'ını alan procedure 'ü yazınız.
def fac(n):
    result=1
    while n>1:
        result=result*n
        n=n-1
    return result

print (fac(3))


# koordinatlarını bildiğimiz (x ve y ) noktalarının üçgen alanını hesaplama
import math
print (math.sqrt(25))
print (math.pow(3,2))

def triangle_area(x1, y1, x2, y2, x3, y3):
    distance1 = math.sqrt(math.pow((x2 - x1),2) + math.pow((y2 - y1),2))
    distance2 = math.sqrt(math.pow((x2 - x3), 2) + math.pow((y2 - y3), 2))
    distance3 = math.sqrt(math.pow((x1 - x3), 2) + math.pow((y1 - y3), 2))
    s = (distance1 + distance2 + distance3) / 2
    area = math.sqrt (s * (s - distance1) * (s - distance2) * (s - distance3))
    return area

print(triangle_area(1,4,5,2,7,9))

