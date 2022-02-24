from channels.generic.websocket import WebsocketConsumer
import json
from time import sleep

class WSConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        
        for i in range(0,1000):
            self.send(json.dumps({"message": f"Hello World{i}"}))
            sleep(1)
            
        
 
 