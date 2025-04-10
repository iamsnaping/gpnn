
from omegaconf import OmegaConf
import omegaconf
import os
def load_config(path='/home/wu_tian_ci/GAFL/configs/model.yaml'):
    config= OmegaConf.load(path)
    return config


def write_file(d,layer,f):
    for key,value in d.items():
        if isinstance(value,omegaconf.DictConfig):
            f.write(key+':'+'\n')
            write_file(value,layer+1,f)
        else:
            f.write(' '*layer+key+':'+str(value)+'\n')

def write_config(config,args,paths):

    dict_name=['lan','fl','vl','atom','knn','gnn','dgc','tse']
    new_dict={}
    for key,value in config.items():
        if key in dict_name:
            continue
        new_dict[key]=value
    val_dict=vars(args)
    for p in paths:
        f=open(p,'w')
        f.write('CONFIG'+'\n')
        write_file(new_dict,0,f)
        f.write('\n\n'+'ARGS'+'\n')
        for key,value in val_dict.items():
            f.write(key+':'+str(value)+'\n')
        f.close()
    