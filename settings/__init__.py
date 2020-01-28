import os

environment = os.environ.get('SPICE_ENV', 'development')

if environment == 'production':
    from .production import *
else:
    from .development import *
