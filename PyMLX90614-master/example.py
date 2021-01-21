from smbus import SMBus
from mlx90614 import MLX90614
import os

if __name__ == "__main__":
    bus = SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    while(1):
        temp=sensor.get_obj_temp()
        print('amb='+str(sensor.get_amb_temp()))
        print('obj='+str(temp))
        i = 0
        if (temp>31):
            i += 1
            if(i == 5):
                i = 0
                os.system('omxplayer -o local /home/pi/Desktop/yinpin.mp3')
        j=0
        while(j<100000):
           j+=1
    bus.close()