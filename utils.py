import os 
from natsort import natsorted
import pickle 

# Get all files in a folder in a sorted order.
def get_files(dir):
    files = natsorted(os.listdir(dir))
    return [os.path.join(dir, i) for i in files]

# fid: file id. 
def get_fid(file):
    return os.path.basename(file).split(".")[0]

# Make a directory. 
def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        
# Write a txt file. 
def write(file_name, data):
    with open(file_name, 'w', encoding='UTF-8') as f:
        f.write(data)

# Read a txt file.
def read(file_name):
    with open(file_name, 'r', encoding = 'UTF-8') as f:
        data = f.read()
    return data

def save_pkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
        
def load_pkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data