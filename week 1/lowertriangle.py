n = int(input("Enter the number of rows : "))
for i in range(n,0,-1):
 for j in range(0,n-1):
   print(" ",end = " ")
   for j in range(n):
     print("*",end=" ")
     print()

