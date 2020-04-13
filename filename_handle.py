import uuid
import os

def temp_filename(extention=''):
    uuid_str = uuid.uuid4().hex
    tmp_file_name = 'tmpfile_%s.%s' % (uuid_str, extention)
    return tmp_file_name

def get_paths():
    BASE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.normpath(os.path.join(BASE_PATH,'..', 'model'))
    DATA_PATH = os.path.normpath(os.path.join(BASE_PATH,'..', 'data'))
    return BASE_PATH, MODEL_PATH, DATA_PATH