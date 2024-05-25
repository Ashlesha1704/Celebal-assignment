def upper_triangle(n):
    for i in range(n):
        for j in range(i + 1):
            print("*", end=" ")
        print()

# Take input from user
n = int(input("Enter the number of rows: "))
upper_triangle(n)
