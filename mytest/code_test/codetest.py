import json
import sys
sys.path.append('/home/wu_tian_ci/GAFL')

from myutils.data_utils import (
    add_weight_decay,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    save_checkpoint,
    getTimeStamp,
    MyEvaluatorActionGenome,
)
'''
        for ind in range(idxes_.shape[0]):
            sa_dict[idxes_[ind].detach().item()]=[
                t_pred[ind].cpu().detach().tolist(),
                t_labe[ind].cpu().detach().tolist(),
                m_pred[ind].cpu().detach().tolist(),
                m_labe[ind].cpu().detach().tolist(),
                c_pred[ind].cpu().detach().tolist(),
                c_labe[ind].cpu().detach().tolist(),
                p_pred[ind].cpu().detach().tolist(),
                p_labe[ind].cpu().detach().tolist()
            ]
'''
path1='/home/wu_tian_ci/GAFL/mytest/code_test/multi.json'
path2='/home/wu_tian_ci/GAFL/mytest/code_test/single.json'
js1=json.load(open(path1,'r'))
js2=json.load(open(path2,'r'))
keys=list(js1.keys())
breakpoint()
data1=js1[0]
data2=js2[0]
for i,j in zip(data1,data2):
    print(i,j)