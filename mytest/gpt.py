class GPNN4(nn.Module):

    def __init__(self, config,layer):
        super().__init__()
        self.layer=layer

        self.gpnn=nn.ModuleList()
        for i in range(self.layer):
            self.gpnn.append(GPNNCell4(config))
        # self.pj=MLPs(config.dims,config.dropout,config.eps,config.gpnn.pj.layer)
        self.edges=[]   

    def visual(self):
        for layer in self.gpnn:
            self.edges.append(layer.edges)
        return self.edges


    def set_visual(self,flag=True):
        for layer in self.gpnn:
            layer.visual=flag

    def forward(self,node_features,obj_feature,edge_feature,prompt=None,task_id=None,mask=None,tfm_mask=None):
        if tfm_mask is not None:
            tfm_mask=torch.cat([tfm_mask[:,:,1:],tfm_mask[:,:,1:]],dim=-1)
            tfm_mask=einops.rearrange(tfm_mask,'b f n -> (b n) f')
            tfm_mask[torch.all(tfm_mask==True,dim=-1)]=False
        for layer in self.gpnn:
            if prompt is not None:
                t_node=torch.cat([node_features,obj_feature],dim=-2)
                t_node=prompt(t_node,task_id)
                node_features=t_node[:,:,0,:].unsqueeze(-2)
                obj_feature=t_node[:,:,1:,:]
            node_features,obj_feature=layer(node_features,obj_feature,edge_feature,mask,tfm_mask)

        # t_node=self.pj(torch.cat([node_features,obj_feature],dim=-2))
        # return t_node[:,:,0,:].unsqueeze(-2),t_node[:,:,1:,:]
        return node_features,obj_feature

class GPNNCell4(torch.nn.Module):
    def __init__(self,config):
        super(GPNNCell4, self).__init__()
        self.message_fun = MessageFunction('linear_concat')
        self.edge_fun=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims*3,config.dims),nn.GELU(),
                                    nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())

        self.link_fun = LinkFunction('one_edge')

        self.residual=tnn.MessageNorm(learn_scale=True)
        self.norm=nn.Sequential(nn.Linear(config.dims,config.dims),tnn.GraphNorm(config.dims),nn.GELU())

        self.residual_obj=tnn.MessageNorm(learn_scale=True)
        self.norm_obj=nn.Sequential(nn.Linear(config.dims,config.dims),tnn.GraphNorm(config.dims),nn.GELU())

        self.merging=nn.Sequential(nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=768 * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.tfm = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config.gpnn.enc_layer
        )
        self.edges=[]
        self.visual=False

        # self._load_link_fun()
    # edge features [batch frames nodes-1 dims] nodes==edges
    # human feature [batch frames 1 dims]
    # obj features [batch frames nodes-1 dims]
    def forward(self, human_feature,obj_features,edge_features,mask=None,tfm_mask=None):
        B,F,N,D=obj_features.shape
        human_features=human_feature.repeat(1, 1, N, 1)

        # print('obj,human',human_feature.shape,human_features.shape,obj_features.shape)
        # breakpoint()
        tmp_edge=self.edge_fun(torch.cat([torch.cat([human_features,edge_features,obj_features],dim=-1), # human-obj
                                          torch.cat([obj_features,edge_features,human_features],dim=-1)],dim=-2))# obj-human
        # tmp_edge1=self.edge_fun(torch.cat([edge_features,obj_features],dim=-1))
        # tmp_edge2=self.edge_fun(torch.cat([edge_features,human_features],dim=-1))
        # tmp_edge=torch.cat([tmp_edge1,tmp_edge2],dim=-2)
        if tfm_mask is not None:
            tmp_edge=self.tfm(einops.rearrange(tmp_edge,'b f n d -> (b n) f d'),
                            src_key_padding_mask=tfm_mask)
        else:
            tmp_edge=self.tfm(einops.rearrange(tmp_edge,'b f n d -> (b n) f d'))
        tmp_edge=einops.rearrange(tmp_edge,'(b n) f d -> b f n d',b=B,n=N*2)

        # breakpoint()
        if mask is not None:
            weight_edge=self.link_fun(tmp_edge)*mask
        else:
            weight_edge=self.link_fun(tmp_edge)
        if self.visual:
            self.edges.append(weight_edge.cpu().detach())
        node_features=torch.cat([human_features,obj_features],dim=-2)

        m_v = self.message_fun(node_features, node_features, tmp_edge)
        m_v=self.merging(m_v)
        weight_edge=weight_edge.expand_as(m_v)
        # if mask is not None:
        #     breakpoint()
        edge_weighted=(weight_edge*m_v)
        edge_weighted_human=edge_weighted[:,:,:N,:]
        edge_weighted_obj=edge_weighted[:,:,N:,:]
        edge_weighted_human=torch.sum(edge_weighted_human,-2,keepdim=True)
        # print('human',human_feature.shape,edge_weighted_human.shape)
        # sum aggregation
        # node_features=node_features+edge_weighted
        human_feature=self.norm(self.residual(human_feature,edge_weighted_human)+human_feature)
        obj_features=self.norm_obj(self.residual_obj(obj_features,edge_weighted_obj)+obj_features)
        # print('obj,human',human_feature.shape,obj_features.shape)
        return human_feature,obj_features


class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,dropout,eps):
        super().__init__()
        self.lin=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,out_dim),nn.LayerNorm(out_dim,eps=eps),nn.GELU())
    def forward(self,X):
        return self.lin(X)
    
    def get_last_layer(self):
        return self.lin[1].weight


class Linear(nn.Module):
    def __init__(self,in_dim,out_dim,dropout,eps,norm=False):
        super().__init__()
        self.norm=norm
        if norm: 
            self.lin=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,out_dim),nn.LayerNorm(out_dim,eps=eps),nn.GELU())
        else:
            self.lin=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,out_dim),nn.GELU())
    def forward(self,X):
        return self.lin(X)

class MLPs(nn.Module):
    def __init__(self,in_dim,dropout,eps,layer=3):
        super().__init__()
        self.lin=nn.ModuleList()
        for i in range(layer-1):
            self.lin.append(Linear(in_dim,in_dim,dropout,eps))
        self.lin.append(Linear(in_dim,in_dim,dropout,eps,True))
    def forward(self,X):
        for layer in self.lin:
            X=layer(X)
        return X


class MLPCLS(nn.Module):
    def __init__(self,in_dim,out_dim,dropout,eps):
        super().__init__()
        self.layer=FFN(in_dim,eps,in_dim*4,dropout)
        self.norm=nn.Sequential(nn.LayerNorm(in_dim,eps=eps),nn.GELU())
        self.cls=nn.Linear(in_dim,out_dim)
    
    def forward(self,X):

        return self.cls(self.norm(X+self.layer(X)))
class TwoLayer(nn.Module):
    def __init__(self, in_dim,dropout,eps):
        super().__init__()
        self.layer=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(in_dim,in_dim))
        self.norm=nn.Sequential(nn.LayerNorm(in_dim,eps=eps),nn.GELU())
    def forward(self,X):

        return self.norm(X+self.layer(X))

class FFN(nn.Module):
    def __init__(self,in_dim,eps=1e-12,hidden_dim=None,dropout=0.3,norm_=True):
        super().__init__()
        if hidden_dim is None:
            out_dim=in_dim*4
        else:
            out_dim=hidden_dim
        self.out_dim=out_dim
        self.in_dim=in_dim
        self.norm_=norm_
        self.layer=nn.Sequential(nn.Linear(in_dim,out_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(out_dim,in_dim))
        self.norm=nn.Sequential(nn.LayerNorm(in_dim,eps=eps),nn.GELU())
    
    def forward(self,X):
        if self.norm_:
            return self.norm(X+self.layer(X))
        else:
            return X+self.layer(X)
    


class DynamicFlatter(nn.Module):
    def __init__(self,in_dim,
                 dropout=0.1,eps=1e-5):
        super().__init__()
        self.dims=in_dim
        # score net

        # frame-level
        self.pj1_1=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.GELU())
        self.pj1_2=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.GELU())

        # self.pj1_2=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.LayerNorm(in_dim),nn.GELU())
        # video-level
        self.pj2_1=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.GELU())
        self.pj2_2=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.GELU())
        # self.pj2_2=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.LayerNorm(in_dim),nn.GELU())

        # # frame-level
        self.score1=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(in_dim//2,in_dim//4),
                                  nn.GELU(),nn.Linear(in_dim//4,1),nn.Sigmoid())
        # video-level
        self.score2=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(in_dim//2,in_dim//4),
                                  nn.GELU(),nn.Linear(in_dim//4,1),nn.Sigmoid())
        

        #         # frame-level
        # self.score1=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(in_dim//2,in_dim//4),nn.LayerNorm(in_dim//4),
        #                           nn.GELU(),nn.Linear(in_dim//4,1),nn.Sigmoid())
        # # video-level
        # self.score2=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(in_dim//2,in_dim//4),nn.LayerNorm(in_dim//4),
        #                           nn.GELU(),nn.Linear(in_dim//4,1),nn.Sigmoid())
    
    

        # fusion
        self.f1=FFN(in_dim,eps,in_dim*4,dropout)
        # self.f1=MLP(in_dim,in_dim,dropout,eps)

        self.f2=FFN(in_dim,eps,in_dim*4,dropout)
        self.vl_weight=[]
        self.fl_weight=[]
        self.visual=False
    

    def set_visual(self,flag):
        self.visual=flag
    

    def get_visual(self):
        return self.vl_weight,self.fl_weight
    
    def get_last_layer(self):
        return self.f2.weight
        
    
    # batch frame nodes dims
    def forward(self,X):
        b,f,n,d=X.shape
        frame_level=self.pj1_2(self.pj1_1(X)+X)

        # f_global_x=torch.mean(frame_level[:,:,:,:(self.dims//2)],dim=-2,keepdim=True)
        # f_local_x=frame_level[:,:,:,(self.dims//2):]
        # # print('global',f_global_x.shape,f_local_x.shape)
        # frame_level=torch.cat([f_global_x.repeat(1,1,n,1),f_local_x],dim=-1)

        frame_scores=self.score1(frame_level)
        X=X*frame_scores
        X=torch.sum(X,dim=-2)
        # batch frame dims
        f1=self.f1(X)

        video_level=self.pj2_2(self.pj2_1(f1)+f1)

        # v_global_x=torch.mean(video_level[:,:,:(self.dims//2)],dim=-2,keepdim=True)
        # v_local_x=video_level[:,:,(self.dims//2):]
        # video_level=torch.cat([v_global_x.repeat(1,f,1),v_local_x],dim=-1)

        video_scores=self.score2(video_level)
        f1=f1*video_scores
        f1=torch.sum(f1,dim=-2)
        f2=self.f2(f1)
        if self.visual:
            self.vl_weight.append(video_scores.cpu().detach())
            self.fl_weight.append(frame_scores.cpu().detach())
        return f2

class ProjectionHeadFT(nn.Module):
    def __init__(self,in_dim,eps=1e-12,hidden_dim=None,dropout=0.3,norm_=True):
        super().__init__()
        if hidden_dim is None:
            out_dim=in_dim*4
        else:
            out_dim=hidden_dim
        self.out_dim=out_dim
        self.in_dim=in_dim
        self.norm_=norm_
        self.layer=nn.Sequential(nn.Linear(in_dim,out_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(out_dim,in_dim))
        self.norm=nn.Sequential(nn.LayerNorm(in_dim,eps=eps),nn.GELU())
    
    def forward(self,X):
        return self.norm(X+self.layer(X))


class GPFPlus(nn.Module):
# flag =true -> nn.embeding else linear
    def __init__(self, config,flag=False):
        super().__init__()
        self.pnums=config.finetune.p_nums
        self.dims=config.dims
        self.flag=flag
        if flag:
            self.tokens=nn.Embedding(config.cls.ag,config.dims*config.finetune.p_nums)
            self.ptokens=nn.Embedding(config.cls.ag,config.dims)
        else:
            self.tokens=nn.Linear(config.cls.ag,config.dims*config.finetune.p_nums)
            self.ptokens=nn.Linear(config.cls.ag,config.dims)
        # global:1 human:1 obj:9
        self.net=nn.Linear(config.dims,config.finetune.p_nums,nn.Sigmoid())

    def get_prompt(self,X,task_id,detach=False):
        b,f,n,d=X.shape
        # batch rels
        task_id=task_id.unsqueeze(1).unsqueeze(1).repeat(1,f,n,1)
        # breakpoint()
        if self.flag:
            task_token=self.ptokens(task_id)
            # task_token=self.p_linear(torch.sum(task_token,dim=-2))+X
            task_token=torch.sum(task_token,dim=-2)+X
            # breakpoint()
        else:
            task_token=self.ptokens(task_id)+X
        # batch frames nodes 1 p_nums
        # weight=F.softmax(self.net(task_token).unsqueeze(-2),dim=-1)
        weight=self.net(task_token).unsqueeze(-2)
        # breakpoint()
        # batch frames nodes p_nums dims
        if self.flag:
            # prompt=self.t_linear(torch.sum(self.tokens(task_id),dim=-2).reshape(b,f,n,self.pnums,self.dims))
            prompt=torch.sum(self.tokens(task_id),dim=-2).reshape(b,f,n,self.pnums,self.dims)
        else:
            prompt=self.tokens(task_id).reshape(b,f,n,self.pnums,self.dims)
        # breakpoint()
        # prompt=self.tokens(task_id).reshape(b,f,n,self.pnums,self.dims)
        # batch frames nodes 1 dims
        prompt=weight@prompt
        prompt=prompt.squeeze(-2)
        if detach:
            X=X+prompt.detach()
            return prompt.detach()
        else:
            return prompt
        # return X
    def forward(self,X,task_id,detach=False):
        b,f,n,d=X.shape
        # batch rels
        task_id=task_id.unsqueeze(1).unsqueeze(1).repeat(1,f,n,1)
        # breakpoint()
        if self.flag:
            task_token=self.ptokens(task_id)
            # task_token=self.p_linear(torch.sum(task_token,dim=-2))+X
            task_token=torch.sum(task_token,dim=-2)+X
            # breakpoint()
        else:
            task_token=self.ptokens(task_id)+X
        # batch frames nodes 1 p_nums
        # weight=F.softmax(self.net(task_token).unsqueeze(-2),dim=-1)
        weight=self.net(task_token).unsqueeze(-2)
        # breakpoint()
        # batch frames nodes p_nums dims
        if self.flag:
            # prompt=self.t_linear(torch.sum(self.tokens(task_id),dim=-2).reshape(b,f,n,self.pnums,self.dims))
            prompt=torch.sum(self.tokens(task_id),dim=-2).reshape(b,f,n,self.pnums,self.dims)
        else:
            prompt=self.tokens(task_id).reshape(b,f,n,self.pnums,self.dims)
        # breakpoint()
        # prompt=self.tokens(task_id).reshape(b,f,n,self.pnums,self.dims)
        # batch frames nodes 1 dims
        prompt=weight@prompt
        prompt=prompt.squeeze(-2)
        if detach:
            X=X+prompt.detach()
        else:
            X=X+prompt
        return X


class Head(nn.Module):
    def __init__(self, dims,eps,dropout,out_cls):
        super().__init__()
        self.dy=DynamicFlatter(dims,dropout,eps)
        self.head=CLSHead(dims,eps,dropout,out_cls)

    def set_visual(self,flag):
        self.dy.set_visual(flag)

    def get_visual(self):
        return self.dy.get_visual()

    def forward(self,X):
        return self.head(self.dy(X))

    def get_last_layer(self):
        return self.head.get_last_layer()

class MixLayer(nn.Module):
    def __init__(self, config,mlp_flag=False):
        super().__init__()
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.transformer.heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.temporal = nn.TransformerEncoder(
            encoder_layer=temporal_encoder_layer, num_layers=config.mix.tfm_layer
        )

        spatial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.transformer.heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.spatial = nn.TransformerEncoder(
            encoder_layer=spatial_encoder_layer, num_layers=config.mix.tfm_layer
        )
        if mlp_flag:
            self.temporal_mlp=TwoLayer(config.dims,config.dropout,config.eps)
            self.spatial_mlp=TwoLayer(config.dims,config.dropout,config.eps)

        self.frames=config.frames
        self.actors=config.actors+1
        self.mlp_flag=mlp_flag


    # temporal batch*node frame dims -> spatial
    # spatial batch* frame node dims -> temporal
    # mask batch frame node
    def forward(self,temporal_x,spaital_x,mask):

        spaital_x=einops.rearrange(spaital_x,'(b f) n d ->  (b n) f d',f=self.frames,n=self.actors) 

        temporal_x=einops.rearrange(temporal_x,'(b n) f d -> (b f) n d',f=self.frames,n=self.actors)
        if mask is not None:
            tem_mask=einops.rearrange(mask,'b f n -> (b f) n',f=self.frames,n=self.actors)
            spa_mask=einops.rearrange(mask,'b f n ->  (b n) f ',f=self.frames,n=self.actors) 
            spa_mask[torch.all(spa_mask==True,dim=-1)]=False
        
        if self.mlp_flag:
            if mask is None:
                out_s=self.spatial(self.spatial_mlp(temporal_x))
                out_t=self.temporal(self.temporal_mlp(spaital_x))
            else:
                out_s=self.spatial(self.spatial_mlp(temporal_x),
                                src_key_padding_mask=tem_mask)
                out_t=self.temporal(self.temporal_mlp(spaital_x),
                                src_key_padding_mask=spa_mask)      
        else:
            if mask is None:
                out_s=self.spatial(temporal_x)
                out_t=self.temporal(spaital_x)
            else:
                # breakpoint()
                out_s=self.spatial(temporal_x,
                        src_key_padding_mask=tem_mask)
                out_t=self.temporal(spaital_x,
                        src_key_padding_mask=spa_mask)
        # breakpoint()
        return out_t,out_s

class MixBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # st
        self.layer1=MixLayer(config,False)
        # ts
        self.layer2=MixLayer(config,True)

        self.frames=config.frames
        self.actors=config.actors+1
        self.mode=config.mix.mode
        self.tmr_mlp=MLP(config.dims,config.dims,config.dropout,config.eps)
        self.spt_mlp=MLP(config.dims,config.dims,config.dropout,config.eps)


    # out temporal spatial
    # temporal batch*node frame dim
    # spatial batch*frame node dim
    def forward(self,temporal_in,spatial_in,mask):

        layer1_t,layer1_s=self.layer1(temporal_in,spatial_in,mask)

        if self.mode=='mix':
            layer2_t=self.tmr_mlp(einops.rearrange(layer1_s,'(b f) n d -> (b n) f d',f=self.frames,n=self.actors)+layer1_t+temporal_in)
            layer2_s=self.spt_mlp(einops.rearrange(layer1_t,'(b n) f d -> (b f) n d',f=self.frames,n=self.actors)+layer1_s+spatial_in)
        else:
            layer2_t=temporal_in+layer1_t
            layer2_s=spatial_in+layer1_s

        layer3_t,layer3_s=self.layer2(layer2_t,layer2_s,mask)
        return layer3_t,layer3_s

class MixTSE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer=config.mix.layer
        self.tse=nn.ModuleList() 
        for i in range(self.layer):
            self.tse.append(MixBlock(config))
        self.mlp=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.GELU())

        
    # batch frame node dim
    # mask batch frame node
    def forward(self,X,mask=None):
        ba,fr,nod,dim=X.shape
        temporal=einops.rearrange(X,'b f n d -> (b n) f d')
        spatial=einops.rearrange(X,'b f n d -> (b f) n d')
        if mask is not None:
            padding_=torch.zeros((ba,fr,1),dtype=torch.bool).to(mask.device)
            mask=torch.cat([padding_,mask],dim=-1)
        for layer in self.tse:
            temporal,spatial=layer(temporal,spatial,mask)
        temporal=einops.rearrange(temporal,'(b n) f d -> b f n d',b=ba,f=fr,n=nod)
        spatial=einops.rearrange(spatial,'(b f) n d -> b f n d',b=ba,f=fr,n=nod)
        return self.mlp(temporal+spatial)
        # return self.merging(torch.cat([temporal,spatial],dim=-1))



class GPNNMix4(nn.Module):


    def __init__(self, config,flag=False,train_stage=1,pre=False):
        super().__init__()
        self.stage=train_stage
        self.flag=flag
        self.config=config
        self.pre=pre
        print('train_stage',self.stage)
        if self.stage in [3,5]:
            self.model_init1(config)
        elif self.stage in [2,1,4,6,7,8,9,10,11]:
            self.model_init1(config)
            self.model_init2(config)
        else:
            raise NotImplementedError
        self.freeze()
    
    def set_visual(self,flag=True):
        self.gpnn.set_visual(flag)
        self.p_head.set_visual(flag)
        self.c_head.set_visual(flag)
        self.cls_head.set_visual(flag)

    def get_visual(self):
        return self.gpnn.visual(),self.p_head.get_visual(),self.c_head.get_visual(),self.cls_head.get_visual()
    
    def model_init1(self,config):
        self.cls_embed=nn.Embedding(38,768,padding_idx=0)
        self.rel_embed=nn.Linear(30,768)
        self.tse=MixTSE(config)


        self.mffn=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        
        self.m_head=Head(config.dims,config.eps,config.dropout,config.cls.ag)
        self.bbx_linear=nn.Sequential(nn.Linear(4,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.fusion=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        self.edge_fun=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims*3,config.dims),nn.GELU(),
                                    nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)
        self.adapter=CLIPAdapter(config)
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        self.pos = nn.Parameter(torch.zeros(1,16,1,768))

    # stage 2/3
    def model_init2(self,config):
        if config.prompt.type==1:
            print('pgfp')
            self.pgpfp=GPFPlus(config,self.flag)
            self.cgpfp=GPFPlus(config,self.flag)
        elif config.prompt.type==0:
            self.pgpfp=SimplePrompt(config,self.flag)
            self.cgpfp=SimplePrompt(config,self.flag)
        else:
            raise NotImplementedError
        self.gpnn=GPNN4(config,config.gpnn.layer.one)
        # self.m_head2=Head(config.dims,config.eps,config.dropout,config.cls.ag)

        self.mffn2=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        self.mffn3=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        self.p_head=Head(config.dims,config.eps,config.dropout,config.cls.ag+1)
        self.c_head=Head(config.dims,config.eps,config.dropout,config.cls.ag+1)

        # front embedding
        

        self.recs=ReconstructNetwork(config)
        self.total_pj=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        self.gf=GateFusion(config)
        self.cls_head=Head(config.dims,config.eps,config.dropout,config.cls.ag)

    def get_weight(self,p_loss,c_loss):
        if self.stage in [6]:
            p_grad = torch.autograd.grad(p_loss, self.c_head.get_last_layer(), retain_graph=True)[0]
        else:
            p_grad = torch.autograd.grad(p_loss, self.p_head.get_last_layer(), retain_graph=True)[0]
        c_grad = torch.autograd.grad(c_loss, self.c_head.get_last_layer(), retain_graph=True)[0]

        d_weight = torch.norm(c_grad) / (torch.norm(p_grad) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
    
    def freeze(self):
        if self.stage in [1,3,4,5,6,7,10,11]:
            return
        for param in self.parameters():
            param.requires_grad = False
        # 2 total/middle
        # 3 middle backbone only
        train_modules={
            2: [self.total_pj,self.cls_head,self.m_head],
            8: [self.total_pj,self.cls_head,self.m_head],
            9: [self.total_pj,self.cls_head,self.m_head]
            #2: [self.mffn, self.m_head, self.gf, self.cls_head, self.total_pj]
        }

        for module in train_modules.get(self.stage,[]):
            for param in module.parameters():
                param.requires_grad=True




  # single branch
    def forward6(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        task_id=torch.cat([task_id,(~task_id.bool()).float()],dim=0)
        nhuman_obj_feature=torch.cat([human_obj_feature,human_obj_feature],dim=0)
        

        # p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        # c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        pc_f=self.mffn2(self.cgpfp(nhuman_obj_feature,task_id))
        edge_feature=torch.cat([edge_feature,edge_feature],dim=0)
        
        pc_human_feature,pc_obj_feature=self.gpnn(pc_f[:,:,0,:].unsqueeze(-2),pc_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        # pc
        pc_feature=torch.cat([pc_human_feature,pc_obj_feature],dim=-2)
        p_features=pc_feature[B:,:,:,:]

        c_features=pc_feature[:B,:,:,:]


        pc_ans=self.c_head(pc_feature)
        p_ans=pc_ans[B:,:]
        c_ans=pc_ans[:B,:]
        # p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        recs=self.recs(t_node)
        

        return c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans
    # dual branch
    def forward7(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)

        
        task_id2=(~task_id.bool()).float()
        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        c_ans=self.c_head(c_features)
        p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        recs=self.recs(t_node)
        

        return c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans
    # total middle

  # single branch
    def forward8(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        with torch.no_grad():
            # nums =node+1
            B,Frame,Nums,dims=frames.shape
            # breakpoint()
            Nums=Nums
            # bbx_list=bbx_list[:,:,1:,:]
            bbx=self.bbx_linear(bbx_list)
            # breakpoint()
            cls_feature=self.cls_embed(cls)
            rel_feature=self.rel_embed(rel)
            

            pos=self.pos.repeat(B,1,Nums,1)
            adapter_feature=self.adapter(frames)
            # supervised
            # pre_feature=
    
            # frames_features=
            # projection head
            frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
            # breakpoint()

            # total features for a consist edge cls
            human_obj_feature=frames_features[:,:,1:,:]
            # supevised by adapter feature
            # scene graph
            human_obj_feature=human_obj_feature+cls_feature
            #no scenegraph
            # human_obj_feature=human_obj_feature
            human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
            human_features=human_feature.repeat(1,1,Nums-2,1)
            obj_feature=human_obj_feature[:,:,1:,:]
            global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
            edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
    
            # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

            # scene graph
            edge_feature=edge_feature+rel_feature

            human_obj_feature=self.mffn(human_obj_feature)
            task_id=torch.cat([task_id,(~task_id.bool()).float()],dim=0)
            nhuman_obj_feature=torch.cat([human_obj_feature,human_obj_feature],dim=0)
            

            # p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
            # c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
            pc_f=self.mffn2(self.cgpfp(nhuman_obj_feature,task_id))
            edge_feature=torch.cat([edge_feature,edge_feature],dim=0)
            
            pc_human_feature,pc_obj_feature=self.gpnn(pc_f[:,:,0,:].unsqueeze(-2),pc_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


            # pc
            pc_feature=torch.cat([pc_human_feature,pc_obj_feature],dim=-2)
            p_features=pc_feature[B:,:,:,:]

            c_features=pc_feature[:B,:,:,:]

            # p_ans=self.p_head(p_features)
            # rec=
            # t_node_features=
            t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        m_ans=self.m_head(human_obj_feature)

        

        return t_ans,m_ans
    # dual branch
    def forward9(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        with torch.no_grad():
            bbx=self.bbx_linear(bbx_list)
            # breakpoint()
            cls_feature=self.cls_embed(cls)
            rel_feature=self.rel_embed(rel)
            

            pos=self.pos.repeat(B,1,Nums,1)
            adapter_feature=self.adapter(frames)
            # supervised
            # pre_feature=
    
            # frames_features=
            # projection head
            frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
            # breakpoint()

            # total features for a consist edge cls
            human_obj_feature=frames_features[:,:,1:,:]
            # supevised by adapter feature
            # scene graph
            human_obj_feature=human_obj_feature+cls_feature
            #no scenegraph
            # human_obj_feature=human_obj_feature
            human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
            human_features=human_feature.repeat(1,1,Nums-2,1)
            obj_feature=human_obj_feature[:,:,1:,:]
            global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
            edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

            # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

            # scene graph
            edge_feature=edge_feature+rel_feature

            human_obj_feature=self.mffn(human_obj_feature)

            
            task_id2=(~task_id.bool()).float()
            p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
            c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
            
            p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

            c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


            p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

            c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


            # rec=
            # t_node_features=
            t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        m_ans=self.m_head(human_obj_feature)
        

        return t_ans,m_ans
    # total middle
  # single branch
    @torch.no_grad()
    def forward10(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature

        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        task_id=torch.cat([task_id,(~task_id.bool()).float()],dim=0)
        # batch frame node dims
        nhuman_obj_feature=torch.cat([human_obj_feature,human_obj_feature],dim=0)
        

        # p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        # c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        pc_f=self.mffn2(self.cgpfp(nhuman_obj_feature,task_id))
        edge_feature=torch.cat([edge_feature,edge_feature],dim=0)
        #obj: batch*2 frame node-1 dims / human:batch*2 frame 1 dims
        pc_human_feature,pc_obj_feature=self.gpnn(pc_f[:,:,0,:].unsqueeze(-2),pc_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        # pc
        pc_feature=torch.cat([pc_human_feature,pc_obj_feature],dim=-2)
        p_features=pc_feature[B:,:,:,:]

        c_features=pc_feature[:B,:,:,:]


        pc_ans=self.c_head(pc_feature)
        p_ans=pc_ans[B:,:]
        c_ans=pc_ans[:B,:]
        # p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))

        m_ans=self.m_head(human_obj_feature)
        

        return c_ans,p_ans,t_ans,m_ans
    # dual branch
    @torch.no_grad()
    def forward11(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)

        
        task_id2=(~task_id.bool()).float()
        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        c_ans=self.c_head(c_features)
        p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        m_ans=self.m_head(human_obj_feature)
        

        return c_ans,p_ans,t_ans,m_ans
    


    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        forwards=[
                 self.forward6,
                 self.forward7,
                 self.forward8,
                 self.forward9,
                 self.forward10,
                 self.forward11]
        return forwards[self.stage-1](frames,cls,rel,bbx_list,task_id,mask,tfm_mask)