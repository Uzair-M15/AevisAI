import pickle
import decimal

f = open('nn1' , 'rb')
nn = pickle.load(f)

dataset = open('city_day.csv' , 'r')
largest = decimal.Decimal('0.0')
for i in dataset.readlines():
    if not i.startswith('-'):
        data = decimal.Decimal(i.split(',')[1])
        if data > largest :
            largest = data


while True :
    prompt = input("Enter Data : ")
    if prompt == '999':
        break
    else:
        prompt = decimal.Decimal(prompt)/decimal.Decimal(largest)
    output = nn(prompt)
    print(output*decimal.Decimal(largest))