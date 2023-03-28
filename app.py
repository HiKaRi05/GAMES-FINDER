import gid as gid
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array


import os

app = Flask(__name__)

# โหลดโมเดล
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/predict", methods={'POST'})
def predict():
    h1 = int(request.form['รับสถาณการณ์กดดันได้ดี'])
    h2 = int(request.form['ช่างสังเกต'])
    h3 = int(request.form['ชอบทดลอง'])
    h4 = int(request.form['รักอิสระ'])
    h5 = int(request.form['ละเอียดรอบคอบ'])
    h6 = int(request.form['มีความสนใจในความรู้รอบตัว'])
    h7 = int(request.form['มีระเบียบวินัย'])
    h8 = int(request.form['คนที่ทันคน'])
    h9 = int(request.form['ชอบลองผิดลองถูก'])
    h10 = int(request.form['ชอบความท้าทาย'])
    h11 = int(request.form['รักสนุก'])
    h12 = int(request.form['รักการอ่าน'])
    h13 = int(request.form['มีความเป็นตัวของตัวเองสูง'])
    h14 = int(request.form['ขี้ระแวง'])
    h15 = int(request.form['เข้าสังคมเก่ง'])
    h16 = int(request.form['มีความหนักแน่นมั่นคง'])
    h17 = int(request.form['ไม่ชอบเที่ยวโลดโผน'])
    h18 = int(request.form['ไม่ค่อยกล้าแสดงออก'])
    h19 = int(request.form['ใจกว้าง'])
    h20 = int(request.form['ขี้เกรงใจคนอื่น'])


    prediction = model.predict(
        [[h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19,
          h20]])  # ทำนายเป็นแนวเกมส์ เช่น Action, Sport

    #class_names = ['Action', 'Adventure', 'RPG', 'Strategy', 'Simulation', 'Fighting', 'Puzzle', 'Sport']

    prediction_class = str(prediction[0])

    # โหลดรูปภาพ
    img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), prediction_class, '1.jpg')
    try:
        img = load_img(img_path, target_size=(224, 224))
    except IOError:
        return "Unable to open image file"

    # แปลงรูปภาพเป็น array
    img_array = img_to_array(img)

    # ปรับค่าความหมายของ array
    img_array = img_array / 255.0

    prediction_class_response = jsonify({'prediction_class': prediction_class})
    prediction_class_response.headers.add('Access-Control-Allow-Origin', '*')  # อนุญาตให้เว็บไซต์อื่นเรียกใช้ API ได้
    # สร้าง response เพื่อส่งรูปภาพกลับไปยังเว็บไซต์
    response = jsonify({'image': img_array.tolist()})
    response.headers.add('Access-Control-Allow-Origin', '*')  # อนุญาตให้เว็บไซต์อื่นเรียกใช้ API ได้

    # ต้องส่ง prediction_class กลับไปเพื่อให้แสดงผลแนวเกมส์ที่ทำนายได้
    return render_template('index.html', prediction_class=prediction_class, response=response)


if __name__ == "__main__":
    app.run()
