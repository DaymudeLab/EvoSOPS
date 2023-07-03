arr = [[[0 for k in range(6)] for j in range(7)] for i in range(7)]
cnt=0
for n in range(7):
    for j in range(7):
        for i in range(6):
            if (i+j <= n):
                cnt+=1
                arr[n][j][i] = round(min(10**(-i+(j-1)),1),4)

print(arr)
print(cnt)