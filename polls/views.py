from django.shortcuts import render
from polls.forms import ImageForm
from keras.models import load_model
from keras.preprocessing import image
import numpy
classes = ["airplane", "automobile", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck", ]


def image_import_view(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            objImg = form.instance
            objImg.save()
            objImg.result = predict(objImg.image.path)
            form.save()
            return render(request, 'index.html', {'form': form, 'objImg': objImg})
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})


def predict(picturePath):
    model_new = load_model("model.h5")
    myImg = image.load_img(picturePath, target_size=(32, 32))
    myImg = image.img_to_array(myImg)
    myImg = numpy.expand_dims(myImg, axis=0)
    myImg = myImg.reshape(1, 32, 32, 3)
    return classes[numpy.argmax(model_new.predict(myImg), axis=1)[0]]
