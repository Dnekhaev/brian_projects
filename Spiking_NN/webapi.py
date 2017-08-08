#import cv2
from flask import Flask, render_template, Response
#from visual_api import mark_on_image,add_info_field,Face_info
import matplotlib.pyplot as plt
import io
from PIL import Image
from math import sin
import numpy as np
import time
TIMER=0
app = Flask(__name__)
def Start_web(): 
# Вот внутри этой штуки надо будет писать код
# Либо использовать глобальные переменные и передавать значения через аргументы page
    values=[sin(i) for i in np.linspace(0,15,1000)]  # Тестирую на синусе
    for i in range(1000):
        plt.plot(values[0:i*10+10]) 
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg') # Строим граф, инициализируем буфер и пишем туда байты графа
        buf.seek(0)
        plt.clf() # Чистим полотно графа
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.getvalue() + b'\r\n\r\n') # Выкидываем байты буфера по ip адресу из app.run
        buf.close()
        time.sleep(0.1) # Это наша частота обновления, по моим наблюдениям держит где-то 5 фпс
    

@app.route('/')
def page():
    return Response(Start_web(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
# Эта штука открывает порт и транслирует page по localhost:port
# Запускаем прогу из консоли как обычно python webapi.py
# Потом в браузере заходим по localhost:port и там смотрим кадры.
# Прога начинает работу когда открывается окно браузера

    app.run(host='0.0.0.0',port =5003,debug=True) 