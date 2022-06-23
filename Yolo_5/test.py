# import datetime
# import time
# import json
# def get_sec(time_str):
#     """Get seconds from time."""
#     h, m, s = time_str.split(':')
#     s = s.split('.')[0]
#     return int(h) * 3600 + int(m) * 60 + int(s)
# start = datetime.datetime.now()
# formated = start.strftime("%H:%M:%S")
# print(json.dumps(formated, indent=4, default=str))
# time.sleep(3)
# end = datetime.datetime.now()
# duration = end - start
# print(get_sec(str(duration)))


# import pyrebase
# config = {'apiKey': "AIzaSyBnU_-WiH0q9nvVyNZ82DgrMi1RMrSOJQk",
#   'authDomain': "mask-detection-system-d20b3.firebaseapp.com",
#   'databaseURL': "https://mask-detection-system-d20b3-default-rtdb.europe-west1.firebasedatabase.app",
#   'projectId': "mask-detection-system-d20b3",
#   'storageBucket': "mask-detection-system-d20b3.appspot.com",
#   'messagingSenderId': "230441871776",
#   'appId': "1:230441871776:web:a8d46d1314a5b29f805cc5",
#   'measurementId': "G-KN8DXJQW5H"
# }
# from datetime import date
# firebase = pyrebase.initialize_app(config)
# storage = firebase.storage()
# storage.child("Images/img2.jpg").put("roomConfig/yolov5-master/test3.jpg")
# print('here')
# global db
# db = firebase.database()
# import datetime
# date = datetime.date.today()
# print(date.today())
import numpy as np
from PIL import Image

arr = np.random.uniform(size=(3,256,256))*255
print(arr)
img = Image.fromarray(arr.T, 'RGB')
img.save('out.png')