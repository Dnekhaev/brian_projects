#import cv2
from flask import Flask, render_template, Response
#from visual_api import mark_on_image,add_info_field,Face_info
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import os.path
from PIL import Image
from math import sin
import numpy as np
import time
TIMER=0
app = Flask(__name__)
def Start_web_weights(): 
# Вот внутри этой штуки надо будет писать код
# Либо использовать глобальные переменные и передавать значения через аргументы page
    #values=[sin(i) for i in np.linspace(0,15,1000)]  # Тестирую на синусе
    #for i in range(1000):
    while(True):
        #img = plt.imread('./tmp/imshow_input_weights.jpg')
        if 1:
            try:
                img1 = Image.open('./tmp/imshow_input_inverse_weights.jpg')
                #img = Image.open('./tmp/image_H.jpg')
                img2 = Image.open('./tmp/imshow_input_weights.jpg')
                img3 = Image.open('./tmp/imshow_intrinsic_weights.jpg')
            except OSError:
                print('got OSError but DEAL with it')
                time.sleep(0.1)
                continue
            plt.figure(figsize=(20,20))
            plt.subplot(131)
            plt.imshow(img1)
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(img2) 
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(img3)
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='jpg') # Строим граф, инициализируем буфер и пишем туда байты графа
            buf.seek(0)
            plt.clf() # Чистим полотно графа
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.getvalue() + b'\r\n\r\n') # Выкидываем байты буфера по ip адресу из app.run
            buf.close()
            plt.close()
            time.sleep(0.1) # Это наша частота обновления, по моим наблюдениям держит где-то 5 фпс
        else:
            time.sleep(0.1) # Это наша частота обновления, по моим наблюдениям держит где-то 5 фпс
            continue

def Start_web_monitors():
# Вот внутри этой штуки надо будет писать код
# Либо использовать глобальные переменные и передавать значения через аргументы page
    #values=[sin(i) for i in np.linspace(0,15,1000)]  # Тестирую на синусе
    #for i in range(1000):
    while(True):
        #img = plt.imread('./tmp/imshow_input_weights.jpg')
        if 1:
            try:
                img0 = Image.open('./tmp/image_H.jpg')
            except OSError:
                print('got OSError but DEAL with it')
                time.sleep(0.1)
                continue
            plt.figure(figsize=(20, 20))
            #plt.figure()
            plt.axis('off')
            plt.imshow(img0)

            buf = io.BytesIO()
            plt.savefig(buf, format='jpg') # Строим граф, инициализируем буфер и пишем туда байты графа
            buf.seek(0)
            plt.clf() # Чистим полотно графа
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.getvalue() + b'\r\n\r\n') # Выкидываем байты буфера по ip адресу из app.run
            buf.close()
            plt.close()
            time.sleep(0.1) # Это наша частота обновления, по моим наблюдениям держит где-то 5 фпс
        else:
            time.sleep(0.1) # Это наша частота обновления, по моим наблюдениям держит где-то 5 фпс
            continue


@app.route('/weights/')
def page_weights():
    return Response(Start_web_weights(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/monitors/')
def page_monitors():
    return Response(Start_web_monitors(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
# Эта штука открывает порт и транслирует page по localhost:port
# Запускаем прогу из консоли как обычно python webapi.py
# Потом в браузере заходим по localhost:port и там смотрим кадры.
# Прога начинает работу когда открывается окно браузера

    app.run(host='0.0.0.0',port =5003,debug=True) 
