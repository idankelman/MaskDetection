from django.urls import path
from .consumers import WSConsumer

ws_urlpatterns = [
    path('ws/test/',WSConsumer.as_asgi())
]