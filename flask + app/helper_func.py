import cv2
import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import Image
#from matplotlib import pyplot as plt
import pandas as pd
from google.cloud import vision
import io
import os, cv2
from collections import Counter
import sys
import plotly.graph_objects as go
import plotly
import pandasql as ps
import re
import json
import plotly.express as px

def run_main(img):

    def increase_brightness(img, value=600):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    img=increase_brightness(img)

    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    img=enhanced_img

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # invert
    thresh = 255 - thresh

    # apply horizontal morphology close
    kernel = np.ones((100 ,10000), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # get external contours
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # draw contours
    result = img.copy()
    for cntr in contours:
        # get bounding boxes
        pad = 100
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x-pad, y-pad), (x+w+pad, y+h+pad), (0, 0, 255), 4)

    
    cv2.imwrite("_box.jpg",result)

    #filename="images/"+image+"_box.jpg"

    def findHorizontalLines(img):
        #img = cv2.imread(img) 
        img =result
        #convert image to greyscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # set threshold to remove background noise
        thresh = cv2.threshold(gray,30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        
        # define rectangle structure (line) to look for: width 100, hight 1. This is a 
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200,1))
        
        # Find horizontal lines
        lineLocations = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        
        return lineLocations

    lineLocations = findHorizontalLines(img)

    img = result 

    #convert image to greyscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # set threshold to remove background noise
    thresh = cv2.threshold(gray,30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


    # define rectangle structure (line) to look for: width 100, hight 1. This is a 
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (500,1))

    # Find horizontal lines
    lineLocations = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    df_lineLocations = pd.DataFrame(lineLocations.sum(axis=1)).reset_index()
    df_lineLocations.columns = ['rowLoc', 'LineLength']
    df_lineLocations[df_lineLocations['LineLength'] > 50]

    df_lineLocations['line'] = 0
    df_lineLocations['line'][df_lineLocations['LineLength'] > 100] = 1

    df_lineLocations['cumSum'] = df_lineLocations['line'].cumsum()

    df_lineLocations['rowLoc'] = df_lineLocations['rowLoc'].astype('int64')
    df_lineLocations['line'] = df_lineLocations['line'].astype('int64')
    df_lineLocations['cumSum'] = df_lineLocations['cumSum'].astype('int64')
    df_lineLocations['LineLength'] = df_lineLocations['LineLength'].astype('int64')


    query = '''
    select row_number() over (order by cumSum) as SegmentOrder
    , min(rowLoc) as SegmentStart
    , max(rowLoc) - min(rowLoc) as Height
    from df_lineLocations
    where line = 0
    --and CumSum !=0
    group by cumSum;
    '''


    df_SegmentLocations  = ps.sqldf(query, locals())



    def pageSegmentation1(img, w, df_SegmentLocations): 
        im2 = img.copy()
        segments = []

        for i in range(len(df_SegmentLocations)):
            y = df_SegmentLocations['SegmentStart'][i]
            h = df_SegmentLocations['Height'][i]

            cropped = im2[y:y + h, 0:w] 
            segments.append(cropped)
            #plt.figure(figsize=(8,8))
            #plt.imshow(cropped)
            #plt.title(str(i+1))        

        return segments

    w = lineLocations.shape[1]
    segments = pageSegmentation1(img, w, df_SegmentLocations)


    new_segments=[]

    for element in segments:
        if len(np.unique(element))>150:
            new_segments.append(element)

    segments=new_segments


    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/pedroleal/Desktop/Ironhack/Final project/Project/Cloudvision/HandwritingRecognition_GoogleCloudVision/my-key.json"
    print('Credendtials from environ: {}'.format(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')))

    def CloudVisionTextExtractor(handwritings):
        # convert image from numpy to bytes for submittion to Google Cloud Vision
        _, encoded_image = cv2.imencode('.png', handwritings)
        content = encoded_image.tobytes()
        #image = vision.types.Image(content=content)
        image = vision.Image(content=content)
        
        # feed handwriting image segment to the Google Cloud Vision API
        client = vision.ImageAnnotatorClient()
        response = client.document_text_detection(image=image)
        
        return response

    def getTextFromVisionResponse(response):
        texts = []
        for page in response.full_text_annotation.pages:
            for i, block in enumerate(page.blocks):  
                for paragraph in block.paragraphs:       
                    for word in paragraph.words:
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        texts.append(word_text)

        return ' '.join(texts)


    def most_frequent(List):
        occurence_count = Counter(List)
        return occurence_count.most_common(1)[0][0].replace(" ","")

    final_equations=[]


    for j in range(0,len(segments)):
        
        first_element=[]
        second_element=[]
        third_element=[]

        for i in range(1,11):

            kernel = np.ones((5, 5), np.uint8)
            img_dilation = cv2.dilate(cv2.bitwise_not(segments[j]), kernel, iterations=i)
            handwritings=cv2.bitwise_not(img_dilation)
            response = CloudVisionTextExtractor(handwritings)
            handwrittenText = getTextFromVisionResponse(response)
            if len(handwrittenText.split("+"))==1:
                first_element.append(handwrittenText.split("+")[0].strip().lower())
                a="first"
            elif len(handwrittenText.split("+"))==2:
                first_element.append(handwrittenText.split("+")[0].strip().lower())
                second_element.append(handwrittenText.split("+")[1].strip().lower())
                a="second"
            elif len(handwrittenText.split("+"))==3:
                first_element.append(handwrittenText.split("+")[0].strip().lower())
                second_element.append(handwrittenText.split("+")[1].strip().lower())
                third_element.append(handwrittenText.split("+")[2].strip().lower())
                a="third"

            else: "Program only works until functios with 3 variables"

        if a=="first":
            final_equations.append(most_frequent(first_element))
        elif a=="second":
            final_equations.append(most_frequent(first_element)+"+"+most_frequent(second_element))
        else: final_equations.append(most_frequent(first_element)+"+"+most_frequent(second_element)+"+"+most_frequent(third_element))


    while "" in final_equations:
        final_equations.remove("")
        
    numbers=[]
    coefs=[[]]*len(final_equations)


    for i in range(0,len(final_equations)):
        #el=final_equations[i].split("=")
        el=re.split('==|=', final_equations[i])
        if el[0]==el[0].split("+"):
            numbers.append(re.findall(r'\d+', el[0])[0])
            a=el[1].split("+")
            a=" ".join(a)
            coefs[i]=re.findall(r'\d+', a)
            numbers[i]=re.findall(r'\d+', numbers[i])
            

        else:
            numbers.append(re.findall(r'\d+', el[1])[0])
            a=el[0].split("+")
            a=" ".join(a)
            coefs[i]=re.findall(r'\d+', a)


    if len(coefs)>len(numbers):
        sys.exit("System underdeterminated. You have more unknowns than equations. You must have the same number.")

    if len(numbers)>len(coefs):
        sys.exit("System overdeterminated. You have more equations than unknowns. You must have the same number.")




    for i in range(0,len(numbers)):
        numbers[i]=int(numbers[i])


    print(coefs)
    print(numbers)

    for i in range(0,len(coefs)):
        for j in range(0,len(coefs)):
            coefs[i][j]=int(coefs[i][j])


    A = np.array(coefs)
    b = np.array(numbers)
    x = np.linalg.solve(A, b)


    if len(x)==1:
        final_result="The solution for the system of equations presented is: \n x= {:.2f}".format(x[0])
    if len(x)==2:
        final_result="The solution for the system of equations presented is: \n x= {:.2f} \n y= {:.2f}".format(x[0],x[1])
    if len(x)==3:
        final_result="The solution for the system of equations presented is: \n x= {:.2f} \n y= {:.2f} \n z= {:.2f}".format(x[0],x[1],x[2])

    solutions=x


    if len(x)==3:
        a=10
        xx= np.arange(round(x[0])-a,round(x[0])+a)
        yy = np.arange(round(x[1])-a,round(x[1])+a)

        cmap = plt.get_cmap("tab10")
        make_int = np.vectorize(int)

        mycolors_a = make_int(256*np.array(cmap(1)[0:3])).reshape((1, 1,-1)).repeat(21, axis = 0).repeat(21, axis =1)
        mycolors_b = make_int(256*np.array(cmap(2)[0:3])).reshape((1, 1,-1)).repeat(21, axis = 0).repeat(21, axis =1)


        #plane 1

        zz1=[]
        for element in xx:
            zz1.append( (numbers[0]- np.ones(len(yy))*element*coefs[0][0] - yy*coefs[0][1]) / coefs[0][2] )

        plane1 = go.Surface(x=xx, y=yy, z=np.array(list(zz1)), surfacecolor = mycolors_a, opacity = .7, showscale = False, name = final_equations[0])

        # plane 2

        zz2=[]
        for element in xx:
            zz2.append( (numbers[1]- np.ones(len(yy))*element*coefs[1][0] - yy*coefs[1][1]) / coefs[1][2] )

        plane2 = go.Surface(x=xx, y=yy, z=np.array(list(zz2)), surfacecolor = mycolors_a, opacity = .7, showscale = False, name = final_equations[1])


        # plane 3

        zz3=[]
        for element in xx:
            zz3.append( (numbers[2]- np.ones(len(yy))*element*coefs[2][0] - yy*coefs[2][1]) / coefs[2][2] )

        plane3 = go.Surface(x=xx, y=yy, z=np.array(list(zz3)), surfacecolor = mycolors_a, opacity = .7, showscale = False, name = final_equations[2])

        figure = go.Figure()
        figure.add_traces([plane1])
        figure.add_traces([plane2])
        figure.add_traces([plane3])
        figure.add_scatter3d(x=[solutions[0]], y=[solutions[1]],z = [solutions[2]], mode='markers',marker={"colorscale":"reds","color":"red","size":10})
        figure.write_html("templates/graph.html")
        graph=plotly.offline.plot(figure,filename="simple-3d-surface",output_type="div")

        #figure.show()


    if len(x)==2:

        a=10
        xx1=np.arange(x[0]-a,x[0]+a)
        yy1=(numbers[0]-coefs[0][0]*xx1)/coefs[0][1]

        xx2=np.arange(x[1]-a,x[1]+a)
        yy2=(numbers[1]-coefs[1][0]*xx2)/coefs[1][1]

        fig = go.Figure()

        line1=go.Scatter(x=xx1, y=yy1)
        line2=go.Scatter(x=xx2, y=yy2)
        point=go.Scatter(x=np.array(solutions[0]),y=np.array(solutions[1]))

        fig.add_trace(line1)
        fig.add_trace(line2)
        fig.add_trace(point)
        fig.write_html("templates/graph.html")
        fig.write_html()
        graph=plotly.offline.plot(fig,filename="simple-2d-line",output_type="div")


    if len(x)==1:

        
        y=np.arange(x[0]-5,x[0]+5)
        x=np.ones(len(y))*x[0]

        plt.plot(x,y)
        plt.legend(final_equations)
        #plt.show()


    return final_result,graph