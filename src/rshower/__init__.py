"""
Radio shower package
"""
__version__ = "0.1.2"
__author__ = "Colley Jean-Marc"

PATH_MODEL = ""


def set_path_model_du(s_path):
    global PATH_MODEL
    PATH_MODEL = s_path


def get_path_model_du():
    return PATH_MODEL
