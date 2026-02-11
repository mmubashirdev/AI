
def add(n1,n2):
  print(f"\n{n1} + {n2} = ",n1+n2)

def sub(n1,n2):
  print(f"\n{n1} - {n2} = ",n1-n2)

def mul(n1,n2):
  print(f"\n{n1} * {n2} = ",n1*n2)

def div(n1,n2):
 print(f"\n{n1} / {n2} = ",n1/n2)


try:
    num1 = int(input("Enter first number: "))
    op = input("Operator +,-,*,/: ")
    num2 = int(input("Enter second number: "))
except ValueError as e:
  print("Enter valid numbers")

else: 
  match op:
    case "+":
      add(num1,num2)
    case "-":
      sub(num1,num2)
    case "*":
      mul(num1,num2)
    case "/":
      if num2 == 0:
        print("Divisor cannot be zero")
      else:
        div(num1,num2)
        
    case _:
      print("Invalid operation")
      
    