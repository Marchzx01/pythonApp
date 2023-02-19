time = float(input("TIME IN: "))
time1 = float(input("TIME OUT: "))
a = time1-time
b = int(a)
print(f'เวลาจอด:{b} ชั่วโมง')
if b <= 0.30:
    print("FREE")
elif b >= 0.31 and b <= 2.59:
    print(f'{b*40} บาท')
elif b >= 3.00 and b <= 5.59:
    print(f'{int(b*100)} บาท')
elif b == 6.00:
    print("500 บาท")
else:
    print("2000 บาท")
