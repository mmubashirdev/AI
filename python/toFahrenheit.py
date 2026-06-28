celsius = float(input("Enter temperature in celsius: "))

def toFahreheit(cel):
  return (cel*9/5)+32

print(f"{celsius} degree into fahrenheit is: ",toFahreheit(celsius)," F")
