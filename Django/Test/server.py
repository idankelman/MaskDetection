import socket,cv2, pickle,struct
import pyshine as ps # pip install pyshine
import json



 
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '10.0.0.6' # Here according to your server ip write the address

port = 9999
client_socket.connect((host_ip,port))
i=0 

if client_socket:
	while (True):
		try:
			data = json.dumps({"message": f"Hello World{i}"})
			client_socket.sendall(data)
			print(f"Sending data: {data}")
			key = cv2.waitKey(1) & 0xFF
			i+=1
			if key == ord("q"):
				client_socket.close()
		except:
			print('VIDEO FINISHED!')
			break