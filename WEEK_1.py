#PYHCHARM1 = PYHCHARM
#KURULUM YAP COOMUNITY EDITION YUKLEYEBILIRIZ.
#Find max of 3 numbers uygulama 1
a = 8
b = 4
if a == b:
    print ( "a is equal to " , b )
else:
    print ( "a is not equal to " , b )

# find max deger of 3 numbers
a=8
b=4
c=5
if  a>b and a>c:
  print ("a is max ",a)
elif b>a and b>c:
 print ("b is max ",b)
elif  c>a and c>b:
 print ("c is max ",c)
# ancak tuning versiyon bu olmalı
def biggest(a,y,z):
    Max=a
    if y>Max:
        Max=y
        if z>Max:
            Max=z
            return Max

# hem 2 ye hem de 3 e bölünebilen sayıları bulunuz
a=8
if  a%6==0 :
    print ("a altıya bölünür ",a)
elif a%2==0 :
    print ("a 2 bölünür ",a)
elif a%3==0 :
   print ("a 3 bölünür ",a)
else:
 print ("hiç birine bölünmez ",a)

# LEN FUNCTION
# Iki string’in uzunlugu ve content’I aynı mıdır?
x="a1c3"
y="a1c3"
if x==y:
    print("content ve uzunluk aynıdır")
elif len(x)==len(y):
    print("uzunlukları aynıdır")
else:
    print("ikisi de değil")

# --GET DATA TYPE OF VARIABLE
var1="a1c3"
var2=3
var3=3.2
print (type(var1))
print (type(var2))
print (type(var3))

# CONVERT INT TO STR AND TYPE 3rd CHARACTER
x=123467
x=str(x)
print(x[2])
# ikinci yöntem
x=123467
print(str(x) [2])

# BOOLEAN
# Find 2 of numbers which one is true the other false or both of them false
a=True
b=False
if (a and b):
    print ("both are true")
elif(a):
    print("a true and b false ")
elif(b):
    print("b false and a true")
else :
    print("no of them true")

# --IKI DEGERIN YERLERINI DEGISTIRME
x=10
y=20

print(x,y)
z=y
y=x
x=z
print (x,y)
# kolay şekilde yapmak için  SWAP Function kullanimi x,y=y,z
