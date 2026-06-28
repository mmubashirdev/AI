a = 10
b = 10.2
c = 'C'
d = "python"
boolean = True
list = [1,2,3,4]
tuple = (1,2,3,4)
set = {1,2,3,4}
dictionary = {"name": "Mubashir", "age":20} 
y = 3
z = 1+y*9  # 


print(list)
print(tuple[0])

num1 = int(input("Enter first number: "))
num2 = int(input("Enter second number: "))

sum = num1+num2

print(f"Sum of {num1} and {num2} is {sum}")


nums = [10,20,30]
sum = 0
for num in nums:
  sum += num
avg = sum/len(nums) 
print("Average is =>",avg)

radius = float(input("Enter the radius of the circle: "))
pi = 3.14
sqr = pi*radius*radius
print("The area of circle is ",sqr)


n = int(input("Enter a number"))
if n%2==0:
  print(n,"is an even number")
else:
  print(n,"is an odd number")