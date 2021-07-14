from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import tensorflow as tf
from tensorflow import keras  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np

# Create your views here.

img_height, img_width=224,224

Labels={'NORMAL':1,'PNEUMONIA':2,'COVID19':0}
def getCode(label):
   return Labels[label]
def getLabel(n):
   for x,c in Labels.items():
      if n==c:
           return x

# with open('./models/imagenet_classes.json','r') as f:
#     labelInfo=f.read()

# labelInfo=json.loads(labelInfo)

model_graph = tf.compat.v1.Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/healthnomic_model.h5')

def Home(request):
    return render(request,"Home.html")

def Browse(request):
    return render(request,"Browse.html")
def scan(request):
    context={'a':1}
    return render(request,"scan.html",context)
def about_us(request):
    return render(request,"about_us.html")

def predictImage(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    img = tf.keras.preprocessing.image.load_img(testimage, target_size=(img_height, img_width)) 
    x=np.array(img)
    x=x/255
    x=x.reshape(1,img_height, img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)


    # predictedLabel=labelInfo[str(np.argmax(predi[0]))]
    predictedLabel=getLabel(np.argmax(predi))
    context={'filePathName':filePathName,'predictedLabel':predictedLabel}
    return render(request,'scan.html',context)