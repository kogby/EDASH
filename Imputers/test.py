import sys
n,m = map(int, input().split())
items = {}
t_list = []
for i in range(n):
    item, price = input().split(" ")
    items[item] = int(price)
for j in range(m):
    transaction = input().split(" ")
    t_list.append(transaction)
    
for line in sys.stdin:
    command = input().split(" ")
    if command[0] == 'A':
        count = 0
        for t in t_list:
            if t[0] == command[1]:
                count += items[t[1]]*int(t[2])
        print(count, '\n')
    elif command[0] == 'B':
        buy = 0
        for t in t_list:
            if t[0] == command[1]:
                if t[1] == command[2]:
                    buy += int(t[2])
        print(buy , '\n')
    else:
        cus_list = []
        for t in t_list:
            if command[1] == t[1]:
                cus_list.append(t[0])
        if not cus_list:
            print(0)
        else:
            print(",".join(cus_list),'\n')
    
