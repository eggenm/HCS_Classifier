step = 4
end = 18
x = range(3, end, step)
for n in x:
    #y = n / 1000
    y=min(n+step, end)
    print(n, y)