from crypt import methods
from flask import Flask, render_template, request, url_for
import os
from helper_func import run_main
import cv2

UPLOAD_FOLDER = "./upload"
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/',methods = ['POST','GET'])
def iris_pred():

    if request.method == 'POST':

        if "file1" not in request.files:
            return "there is no file1 in form!"
        file1 = request.files["file1"]
        
        path = os.path.join(app.config["UPLOAD_FOLDER"], file1.filename)
        file1.save(path)
        
        print(file1.filename)
        print(type(file1.filename))

        img=cv2.imread("upload/"+file1.filename)
        

        final_result=run_main(img)
        
        
        return render_template('main.html', pred = final_result )

    else:
        return render_template('main.html')


if __name__ == "__main__":
    app.run(debug = True, port = 8735)

