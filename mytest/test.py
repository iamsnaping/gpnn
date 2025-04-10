import torch
import sys
sys.path.append('/home/wu_tian_ci/GAFL')

from myutils.config import *
import argparse
# a=torch.tensor([[[2,3,4]]])
# b=torch.tensor([[[1,2,3],[2,3,4],[4,5,6]]])
# c=(a@b).squeeze(-2)
# print(a.shape,b.shape)
# print(c)
# a=[12,3]
# b=[2,3,4]
# c=a+b
# print(c)
# a.extend(b)
# print(a)

# a=[[1,2,3],[2,3,4]]
# for l in a:
#     l[2]=1
# print(a)

parser = argparse.ArgumentParser(description="Packs PIL images as HDF5.")

parser.add_argument(
    "--device",
    type=str,
    default="cuda:2",
    help="gpu device",
)
parser.add_argument(
    "--sup",
    type=str,
    default="nothing",
    help="sth to say",
)
parser.add_argument(
    "--epoch",
    type=int,
    default=20,
    help="train epochs",
)
parser.add_argument(
    "--warmup",
    type=int,
    default=2,
    help="warmup epochs",
)
parser.add_argument(
    "--batchsize",
    type=int,
    default=32,
    help="batchsize",
)
parser.add_argument(
    "--lr",
    type=float,
    default=2e-4,
    help="learning rate",
)
parser.add_argument(
    "--decay",
    type=float,
    default=1e-3,
    help="learning rate",
)
parser.add_argument(
        "--clip_val",
        type=float,
        default=5.0,
        help="The gradient clipping value.",
    )
parser.add_argument(
    "--model",
    type=str,
    default="mix3",
    help="model",
)
parser.add_argument(
    "--ds",
    type=int,
    default=2,
    help="dataset",
)
parser.add_argument(
    "--stage",
    type=int,
    default=1,
    help="train stage",
)
parser.add_argument(
    "--tp",
    type=int,
    default=0,
    help="train type 0:oracle 1:oracle continue 2:pure",
)
parser.add_argument(
    "--prompt",
    type=int,
    default=1,
    help="prompt type 0:smiple 1:gpfp",
)
config=load_config()
print(type(config))

args = parser.parse_args()
# f=open('/home/wu_tian_ci/GAFL/test/test.txt','w')


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
# p1='/home/wu_tian_ci/GAFL/test/test1.txt'
# p2='/home/wu_tian_ci/GAFL/test/test2.txt'
# write_config(config,args,[p1,p2])

import numpy as np
np.random.seed(2)
a=np.abs(np.random.randn(5))
b=np.abs(np.random.randn(5))
c=np.abs(np.random.randn(5))
d=np.array([a,b,c])
aa=np.zeros(5)
bb=np.zeros(5)
cc=np.zeros(5)
aa[0],aa[1],aa[2]=1.,1.,1.
bb[2],bb[3],bb[4]=1.,1.,1.
cc[0],cc[1],cc[4]=1.,1.,1.
dd=np.array([aa,bb,cc])
def mm(d,dd):
    m_aps=[]
    for i in range(5):
        sorted_idxs = np.argsort(-d[:, i])
        tp = dd[:, i][sorted_idxs] == 1.
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            # m_aps.append(0.0)
            continue
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs + t_pcs).astype(float)
        avg_prec = 0
        for i in range(d.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        # print(n_pos)
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
# m_ap = np.mean(m_aps)
    m_ap=np.nanmean(m_aps)
    print(m_ap)

print(d)
print(dd)

a1=np.array([a,b])
b1=np.array([b,c])
c1=np.array([c,a])
aa1=np.array([aa,bb])
bb1=np.array([bb,cc])
cc1=np.array([cc,aa])

mm(d,dd)
mm(a1,aa1)
mm(b1,bb1)
mm(c1,cc1)

# b=np.invert(a)
# print(np.cumsum(a))
# print(np.cumsum(b))
# tp=np.cumsum(a)
# fp=np.cumsum(b)
# ap=tp/(tp+fp)
# m=0
# print(ap)
# counter=0
# i=-1
# for aa in a:
#     i+=1
#     if aa==False:
#         continue
#     m+=ap[i]
#     counter+=1
# print(m/counter)
a=[1,1,0,0,1,1]
print(np.cumsum(a))
a=[1]
b=[1,2,3]
c=[]
c.append([5]+b)
c.append([4]+b)
print(c)