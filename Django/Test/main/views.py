from django.shortcuts import render

# Create your views here.
def index(response):
    print("hello")
    ImgUrl = './static/Images/Temp.jpg'
    return render(response, './main/Home.html',{'Img':ImgUrl,})