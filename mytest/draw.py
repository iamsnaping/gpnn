import matplotlib.pyplot as plt
import os





def get_data(p):
    flist=os.listdir(p)
    flist=sorted(flist,key=lambda x:int(x[0]))
    clist,plist,tlist=[i for i in range(20)],[i for i in range(20)],[i for i in range(20)]
    for f in flist:
        ds=f.split('_')
        # print(ds)
        clist[int(ds[0])-1]=float(ds[1][2:])
        plist[int(ds[0])-1]=float(ds[3][2:-4])
        tlist[int(ds[0])-1]=float(ds[2][2:])
    return clist,plist,tlist

# get_data()


def draw_list(p,name):
    # y1,y2,y3=get_data(p)
    y1=[0.011,0.06,0.161,0.301,0.459,0.591,0.714,0.809,0.869,0.916,0.948,0.967,0.98,0.987,0.992,0.996,0.998,0.999,0.999,0.999,0.999,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

    x=[i for i in range(1,len(y1)+1)]
    plt.plot(x, y1, marker='o', linestyle='-', color='b', label='total')

    # # 绘制第二条折线
    # plt.plot(x, y2, marker='s', linestyle='--', color='r', label='private')

    # plt.plot(x, y3, marker='s', linestyle='--', color='g', label='common')

    # 添加标题和标签
    plt.title('mAP')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 添加网格
    # plt.grid(True)

    # 添加图例
    plt.legend()

    plt.savefig(os.path.join(
        '/home/wu_tian_ci/GAFL/test',
        name
    ))



def draw_list_multi(ys,x,name,names):
    t=0
    co=['r','g','b','y']
    for y in ys:
        plt.plot(x,y,marker='s', linestyle='--', color=co[t], label=names[t])
        t+=1

    # 添加标题和标签
    plt.xticks(x) 
    plt.title('mAP')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 添加图例
    plt.legend()

    plt.savefig(os.path.join(
        '/home/wu_tian_ci/GAFL/mytest',
        name
    ))


if __name__=='__main__':
    p='/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250310/1934'
    draw_list(p,'train')