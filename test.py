from AevisTemp import *
import pickle
import decimal

f = open('city_day.csv')

#compile data
data = []
for line in f.readlines():
    if not line.startswith('-'):
        data.append(decimal.Decimal(line.split(',')[1]))

#Find largest
largest = decimal.Decimal('0.0')
for i in data:
    if i > largest :
        largest = i

data_copy = []
for i in data:
    data_copy.append(i/largest)

data = data_copy

#Create io pairs
end = False
i = 0
io = []
while not end:
    if len(data)-i < 5 :
        end = True
        break

    io.append([data[i : i+1] , data[i+1]])
    i += 1

for i in io:
    print('\n')
    for j in i :
        print(str(j))

nn = lstmCell()

for i in range(20):
    print("")
    print("Iteration " , i)
    nn.learn(io)
    pass

f = open("nn1" , 'wb')
f.writelines([b''])
pickle.dump(nn , f)
f.seek(0)
f.close()
