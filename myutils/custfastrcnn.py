
import torch
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union,Callable

from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import (FastRCNNPredictor,_default_anchorgen,TwoMLPHead,FastRCNNConvFCHead)
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import (_default_anchorgen,
                                                      RPNHead,
                                                      FasterRCNN,
                                                      FastRCNNConvFCHead)
from torchvision.models.detection.backbone_utils import (resnet_fpn_backbone,
                                                         _validate_trainable_layers,
                                                         _resnet_fpn_extractor)
from torchvision.models.detection.faster_rcnn import (_default_anchorgen,
                                                      RPNHead,
                                                      FasterRCNN,
                                                      FastRCNNConvFCHead)
import torchvision.models as TVM

class CustomedGeneralizedRCNN(GeneralizedRCNN):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads: nn.Module, transform: nn.Module,oracle=False) -> None:
        super().__init__(backbone,rpn,roi_heads,transform)
        self.fpn_features=None
        self.visual_features=None
        self.oracle=oracle
       


    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        # self.fpn_features=features
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        self.fpn_features=features
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        print('generation',len(detections),detections[0].keys(),detections[0]['labels'].shape,detections[0]['boxes'].shape,detections[0]['scores'].shape)
        # if self.oracle:
        # roi features
        self.visual_features = self.roi_heads.box_head.features


        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

class CustomedFasterRCNN(GeneralizedRCNN):

    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        **kwargs,
    ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        if not isinstance(rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )
        if not isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(box_roi_pool)}"
            )

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            rpn_anchor_generator = _default_anchorgen()
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = MyPredictor(representation_size, num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )
        # image net transform parameters
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        super().__init__(backbone, rpn, roi_heads, transform)
        


class CusFastRCNNConvFCHead(FastRCNNConvFCHead):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        conv_layers: List[int],
        fc_layers: List[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__(input_size,conv_layers,fc_layers,norm_layer)
        self.features=None
    

    def forward(self,x):
        # front means need extra process
        # self.features=x
        for layer in self:
            x=layer(x)
        # back means do not need
        self.features=x
        return x



class MyPredictor(FastRCNNPredictor):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels=in_channels,num_classes=num_classes)
        self.features=''
        self.cls=''
        self.bbx=''

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        self.features=x.clone()
        self.bbx=bbox_deltas.clone()
        self.cls=scores.clone()
        return scores, bbox_deltas

class MyRCNN(nn.Module):
    def __init__(self,n_class=157,backbone='resnet101',trainable_backbone_layers=1):
        super().__init__()
        self.backbone=backbone
        self.n_class=n_class
        self.model=None
        self.trainable_backbone_layers=trainable_backbone_layers
        self.load_model()
    

    def init_param(self):
        # freeze resnet
        for param in self.model.backbone.parameters():
            param.requires_grad=False
    def load_model(self):
        if self.backbone=='resnet101':
            resnet=TVM.resnet101(weights=TVM.ResNet101_Weights.DEFAULT)
        elif self.backbone=='resnet50':
            resnet=TVM.resnet50(weights=TVM.ResNet50_Weights.DEFAULT)
        is_trained = True
        trainable_backbone_layers = _validate_trainable_layers(is_trained, self.trainable_backbone_layers, 5, 3)
        backbone = _resnet_fpn_extractor(resnet, trainable_backbone_layers, norm_layer=nn.BatchNorm2d)
        rpn_anchor_generator = _default_anchorgen()
        rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
        box_head=FastRCNNConvFCHead((backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d)
        
        self.model = CustomedFasterRCNN(
        backbone,
        num_classes=self.n_class,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,)

    def get_fpn(self):
        return self.model.fpn_features

    def get_visual(self):
        return self.model.visual_features
    
    def get_entity(self):
        return self.entity(self.get_visual())
    
    def get_global_relation(self):
        features=[]
        global_features=self.get_fpn()
        for i in range(4):
            features.append(self.global1(global_features.get(str(i))))
        features.append(self.global1(global_features.get('pool')))
        stack_features=torch.cat(features,dim=-1)
        return self.global_relaiton(stack_features)
    # target is must if train
    def forward(self,images,gt_target=None):
        # print(len(images),len(gt_target))
        return self.model(images,gt_target)

class CusTwoLayer(nn.Module):

    def __init__(self,input_dim,output_dim,dropout=0.3):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(input_dim,output_dim),
                                  nn.LayerNorm(output_dim),nn.GELU(),nn.Dropout(dropout),
                                  nn.Linear(output_dim,output_dim),nn.LayerNorm(output_dim),nn.GELU())
    def forward(self,X):
        return self.layers(X.flatten(start_dim=1))


class MyFPN(nn.Module):
    def __init__(self,n_class=500,backbone='resnet101',trainable_backbone_layers=0):
        super().__init__()
        self.backbone=backbone
        self.n_class=n_class
        self.model=None
        self.trainable_backbone_layers=trainable_backbone_layers
        self.load_model()
      
    def init_param(self):
        # freeze resnet
        for param in self.model.backbone.parameters():
            param.requires_grad=False

    def load_model(self):
 
        if self.backbone=='resnet101':
            resnet=TVM.resnet101(weights=TVM.ResNet101_Weights.DEFAULT)
        elif self.backbone=='resnet50':
            resnet=TVM.resnet50(weights=TVM.ResNet50_Weights.DEFAULT)
        is_trained = True
        trainable_backbone_layers = _validate_trainable_layers(is_trained, self.trainable_backbone_layers, 5, 3)
        backbone = _resnet_fpn_extractor(resnet, trainable_backbone_layers, norm_layer=nn.BatchNorm2d)
        self.model=backbone


    # return fpn
    def forward(self,images):
        return self.model(images)


if __name__=='__main__':


    rcnn=MyRCNN(backbone='resnet101')
    rcnn.init_param()
    rcnn.to('cuda:0')
    # net=TVM.resnet101(weights=TVM.ResNet101_Weights.DEFAULT)
    img=torch.randn(1,1,768)
    img2=torch.randn(2,3,224,224).to('cuda:0')
    # rcnn.eval()
    # rcnn.to('cuda:0')
    boxs=torch.tensor([1,2,3,4]).reshape((1,4)).to('cuda:0')
    lables=torch.tensor([1]).to('cuda:0')
    sroces=torch.tensor([0.1]).to('cuda:0')
    data_dict=dict()
    data_dict['boxes']=boxs
    data_dict['labels']=lables
    data_dict['scores']=sroces
    data_list=[data_dict]
    boxs=torch.tensor([[1,2,3,4],[2,3,4,5]]).reshape((2,4)).to('cuda:0')
    lables=torch.tensor([1,2]).to('cuda:0')
    sroces=torch.tensor([0.1,0.2]).to('cuda:0')
    data_dict2=dict()
    data_dict2['boxes']=boxs
    data_dict2['labels']=lables
    data_dict2['scores']=sroces
    data_list.append(data_dict2)

    ans=rcnn(img2,data_list)

    print(rcnn.model.roi_heads.box_predictor.features.shape)
 
    print((len(ans)))
    print(type(ans),ans.keys())
    # fpn=rcnn.get_fpn()
    # print(rcnn.get_global_relation().shape)
    # for key,value in ans.items():
    #     print('key: ',key)
    #     print('value: ',value.shape)
    # print(rcnn.get_visual().shape)