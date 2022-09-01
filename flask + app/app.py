from crypt import methods
from typing import final
from flask import Flask, render_template, request, url_for
import os
from helper_func import run_main
import cv2
import json
import plotly

UPLOAD_FOLDER = "./upload"
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

final_result = ''
graph=""

@app.route('/',methods = ['POST','GET'])
def math_buddy():
    global final_result
    global graph

    if request.method == 'POST':

        if "file1" not in request.files:
            return "there is no file1 in form!"
        file1 = request.files["file1"]
        
        path = os.path.join(app.config["UPLOAD_FOLDER"], file1.filename)
        file1.save(path)
        
        print(file1.filename)
        print(type(file1.filename))

        img=cv2.imread("upload/"+file1.filename)
        

        final_result,graph=run_main(img)
        
        return render_template('main.html',pred=final_result)

    else:
        return render_template('main.html')

@app.route("/test")
def test():
    return render_template('real_graph.html',pred=final_result)

if __name__ == "__main__":
    app.run(debug = True, port = 8734)

