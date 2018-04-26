import os
import sys

def getparentdir():
    pwd = sys.path[0]
    abs_path = os.path.abspath(pwd)
    return abs_path
