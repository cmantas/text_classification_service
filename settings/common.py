import os

DEBUG = False

settings_source = 'common'

saved_models_dir = 'data/saved_models'

# TODO: fix tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
