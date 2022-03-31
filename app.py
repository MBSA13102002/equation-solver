# import tensorflow as tf
import base64
import cv2
import os
import numpy as np
import json
from flask import Flask, render_template, request
from cnn import ConvolutionalNeuralNetwork
from io import BytesIO
# from test import solveEquation
app = Flask(__name__)

def calculate_operation(operation):
    # def operate(fb, sb, op):
    #     if operator == '+':
    #         result = int(first_buffer) + int(second_buffer)
    #     elif operator == '-':
    #         result = int(first_buffer) - int(second_buffer)
    #     elif operator == 'x':
    #         result = int(first_buffer) * int(second_buffer)
    #     return result

    if not operation or not operation[0].isdigit():
        return -1
    result = operation
    # operator = ''
    # first_buffer = ''
    # second_buffer = ''

    # for i in range(len(operation)):
    #     if operation[i].isdigit():
    #         if len(second_buffer) == 0 and len(operator) == 0:
    #             first_buffer += operation[i]
    #         else:
    #             second_buffer += operation[i]
    #     else:
    #         if len(second_buffer) != 0:
    #             result = operate(first_buffer, second_buffer, operator)
    #             first_buffer = str(result)
    #             second_buffer = ''
    #         operator = operation[i]

    # result = int(first_buffer)
    # if len(second_buffer) != 0 and len(operator) != 0:
    #     result = operate(first_buffer, second_buffer, operator)

    return result

@app.route('/')
def index():
    return render_template('temp.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        # print(str(request.form['operation']))
        operation = BytesIO(base64.urlsafe_b64decode(request.form['operation']))
        CNN = ConvolutionalNeuralNetwork()
        operation = CNN.predict(operation)

        return json.dumps({
            'operation': operation,
            'solution': calculate_operation(operation)
        })

@app.route('/newpred', methods=['POST'])
def newpred():
    if request.method == "POST":
        file = request.files['file']
        print(file)
        file.save("bhabhai.png")
        im = cv2.imread("bhabhai.png",0)
        # cv2.imwrite("gh.png",im)
        # im = cv2.imread("gh.png",0)
        im = cv2.resize(im,(600,200))
        ret, thresh = cv2.threshold(im, 127, 255, 0)
        kernel = np.zeros((3,3),np.uint8)
        # thresh = ~thresh
        thresh = cv2.dilate(thresh,kernel,iterations = 1)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # thresh =~thresh
        thresh = thresh[20:,5:]
        retval, buffer_img= cv2.imencode('.png', thresh)
        data = base64.b64encode(buffer_img)
        data = str(data)
        data = str(data)[2:len(data)-1]
        operation = BytesIO(base64.urlsafe_b64decode(data))
        CNN = ConvolutionalNeuralNetwork()
        operation = CNN.predict(operation)

        # os.remove("./bhaibhai.png")
        return str(operation)

        # # print(str(request.form['operation']))
        # operation = BytesIO(base64.urlsafe_b64decode(request.form['operation']))
        # CNN = ConvolutionalNeuralNetwork()
        # operation = CNN.predict(operation)

        # return json.dumps({
        #     'operation': operation,
        #     'solution': calculate_operation(operation)
        # })

# if __name__ == "__main__":
#     app.run(debug=True)
