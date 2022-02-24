"""
ASGI config for Test project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/asgi/
"""

#===============================================================================
#                Django Imports for the Web Server
#===============================================================================

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter
from channels.routing import URLRouter

# from channels.auth import AuthMiddlewareStack
# import Test.routing as routing
# from Test.routing import ws_urlpatterns


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Test.settings')


application =ProtocolTypeRouter({
    'http': get_asgi_application(),
    # 'websocket':AuthMiddlewareStack(URLRouter(routing.ws_urlpatterns))
})


