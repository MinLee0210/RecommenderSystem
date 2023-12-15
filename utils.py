import yaml
import os

# ----- Read file -----
def yaml_read(dir): 
    content = yaml.load(open(dir, 'r'), Loader=yaml.FullLoader)
    return content

def yaml_write(dir, data): 
    yaml.dump(data, open(dir, 'w'))
    
def make_route(src, tgt):
    if not os.path.isdir(src): 
        return "Invalid directory or the directory is not exist"
    route = os.path.join(src, tgt)
    return route