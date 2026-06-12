# ------------------------------------------------------------------------
#  Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]

import torch
import torch.nn.functional as F
from torch import device, nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
import copy
from ..deformable_detr.ops.modules import (RestrictedMSDeformAttn,MSDeformAttn)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,restrict_attn = True,contrastive=False):
        super().__init__()
        self.contrastive = contrastive
        # cross attention
        if restrict_attn:
            self.cross_attn = RestrictedMSDeformAttn(d_model, n_levels, n_heads, n_points)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, attn_mask=None):
        # self attention
        if self.contrastive:
            q = k = self.with_pos_embed(tgt, None)
        else:
            q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),attn_mask=attn_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class Decouple_DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self,detr,decouple_decoder,decouple_classes,n_heads=8,training_type='sequential',boxes_mode=1,cls_num = 50):
        super().__init__()
        '''
        detr: the original deformable DETR model
        decouple_decoder: the decouple decoder module
        decouple_classes: number of classes for decouple decoder
        training_type: 'sequential' or 'parallel' training
        boxes_mode:
            1: don't add noise to boxes
            2: only add positive noise to boxes
            3: DINO strategy: add both positive and negative noise to boxes
        '''
        self.detr = detr
        assert training_type in ['sequential','parallel','contrastive'], f"training_type should be sequential, parallel or contrastive, but got {training_type}"
        self.training_type = training_type
        self.hook = self.detr.transformer.encoder.register_forward_hook(self.encode_hook)
        self.decouple_decoder = decouple_decoder
        self.num_head = n_heads
        self.boxes_mode = boxes_mode
        hidden_dim = self.detr.transformer.d_model
        self.decouple_class_embed = nn.Linear(hidden_dim, decouple_classes)
        self.wh_project = nn.Linear(2, hidden_dim)  # for initialing query
        self.xy_project = nn.Linear(2, hidden_dim) # for initialing positionl embedding
        if self.training_type == 'contrastive':
            self.contrastive_project = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                                     nn.ReLU(),
                                                     nn.Linear(hidden_dim*2, int(hidden_dim/2)))
        self.cls_num = cls_num
    def forward(self, src: NestedTensor,targets=None,stage='train',threshold=0.4):
        if self.training_type == 'sequential':
            return self.forward_sequential(src)
        elif self.training_type == 'parallel':
            return self.forward_parallel(src,targets,stage,threshold)
        elif self.training_type == 'contrastive':
            return self.forward_contrastive(src,targets,stage,threshold)
        else:
            raise NotImplementedError(f"Training type {self.training_type} is not implemented yet.")
    def forward_sequential(self, src: NestedTensor):
        '''
        Sequential training: only use the boxes from DETR output, the classify module has same sequence length as DETR output.
        '''
        if not isinstance(src, NestedTensor):
            src = nested_tensor_from_tensor_list(src)
        detr_output = self.detr(src)
        boxes = detr_output['pred_boxes'].detach().clone()
        init_query = self.wh_project(boxes[:,:,2:])
        init_pos = self.xy_project(boxes[:,:,:2])
        memory = self.encoder_output
        spatial_shapes,level_start_index,valid_ratios,_,mask_flatten = self.encoder_input[1:]
        decouple_output = self.decouple_decoder(init_query,boxes,memory,spatial_shapes,level_start_index,valid_ratios,init_pos,mask_flatten,)
        class_logits = self.decouple_class_embed(decouple_output)
        detr_output['decouple_class_logits'] = class_logits
        return detr_output
    def forward_parallel(self, src: NestedTensor,targets,stage,threshold):
        if not isinstance(src, NestedTensor):
            src = nested_tensor_from_tensor_list(src)  
        detr_output = self.detr(src)
        boxes,length_label,attention_mask = self.get_boxes_attn_mask(targets,detr_output,stage,threshold)
        init_query = self.wh_project(boxes[:,:,2:])
        init_pos = self.xy_project(boxes[:,:,:2])
        memory = self.encoder_output
        spatial_shapes,level_start_index,valid_ratios,_,mask_flatten = self.encoder_input[1:]
        decouple_output = self.decouple_decoder(init_query,boxes,memory,spatial_shapes,level_start_index,valid_ratios,init_pos,mask_flatten,attention_mask)
        class_logits = self.decouple_class_embed(decouple_output)
        detr_output['decouple_class_logits'] = class_logits
        detr_output['instance_length'] = length_label
        detr_output['instance_boxes'] = boxes
        return detr_output
    def forward_contrastive(self,src: NestedTensor,targets,stage,threshold=0.4):
        if not isinstance(src, NestedTensor):
            src = nested_tensor_from_tensor_list(src)  
        detr_output = self.detr(src)
        boxes,length_label,attention_mask = self.get_boxes_attn_mask_contrastive(targets,detr_output,stage,threshold)
        init_query = self.wh_project(boxes[:,:,2:])
        init_pos = self.xy_project(boxes[:,:,:2])
        memory = self.encoder_output
        spatial_shapes,level_start_index,valid_ratios,_,mask_flatten = self.encoder_input[1:]
        decouple_output = self.decouple_decoder(init_query,boxes,memory,spatial_shapes,level_start_index,valid_ratios,init_pos,mask_flatten,attention_mask)
        contrastive_logits = self.contrastive_project(decouple_output)
        detr_output['contrastive_logits'] = contrastive_logits
        detr_output['instance_length'] = length_label
        detr_output['instance_boxes'] = boxes
        return detr_output
    def encode_hook(self,module,input,output):
        self.encoder_output = output
        self.encoder_input = input
    def get_boxes_attn_mask(self,targets,detr_ouput,stage,threshold=0.4):
        batch_size = detr_ouput['pred_boxes'].shape[0]
        padding_box = torch.tensor([0.5,0.5,1,1],device=detr_ouput['pred_boxes'].device)
        if stage in ['train','eval']:      # using groundtruth boxes during training but predicted boxes during inference
            boxes = [target['boxes'] for target in targets]
            length_label = [box.shape[0] for box in boxes]
            max_length = max(length_label)
            if self.boxes_mode == 2:  # adding noise to groundtruth boxes
                boxes_noised = get_noised_boxes(scale_n_l=0.1,scale_n_u=0.3,positive_negative_ratio=0.5,ground_truth=torch.concat(boxes,dim=0),noise_percentage=0.6)
                boxes = []
                index = 0 
                for l in length_label:
                    boxes.append(boxes_noised[index:index+l,:])   # keep same format
                    index += l 
            if self.boxes_mode == 3: # using dino strategy to add positive and negative boxes during training
                boxes_gt = torch.cat(boxes,dim=0)
                boxes_positive = get_noised_boxes(scale_n_l=0.1,scale_n_u=0.3,positive_negative_ratio=0.5,ground_truth=boxes_gt,noise_percentage=0.8)
                boxes_negative = get_noised_boxes(scale_n_l=0.6,scale_n_u=0.9,positive_negative_ratio=0.5,ground_truth=boxes_gt,noise_percentage=1)
                boxes = []
                index = 0
                for l in length_label:
                    boxes.append(torch.concat([boxes_gt[index:index+l,:],boxes_positive[index:index+l,:],boxes_negative[index:index+l,:]],dim=0))   # keep same format
                    index += l   
            attention_mask = get_attention_mask(length_label,self.num_head,device='cuda',boxes_mode=self.boxes_mode,cls_num=self.cls_num)
            boxes_padded = padding_box.repeat(batch_size,max_length+self.cls_num,1) if self.boxes_mode != 3 else padding_box.repeat(batch_size,3*max_length+self.cls_num,1)
        elif stage in ['test','inference']:
            boxes = detr_ouput['pred_boxes'].detach().clone()
            instance_labels = detr_ouput['pred_logits'].sigmoid()>threshold
            length_label = [len(torch.where(label)[0]) for label in instance_labels]
            max_length = max(length_label)
            attention_mask = get_attention_mask(length_label,self.num_head,device='cuda',boxes_mode=self.boxes_mode,cls_num=self.cls_num)
            boxes_padded = padding_box.repeat(batch_size,max_length+self.cls_num,1)
        for b in range(batch_size):
            if stage in ['train','eval']:
                box = boxes[b]
            else:
                box = boxes[b][instance_labels[b].squeeze(-1)]
            boxes_padded[b,:len(box),:] = box
        return boxes_padded,length_label,attention_mask
    def get_boxes_attn_mask_contrastive(self,targets,detr_ouput,stage,threshold=0.4,global_cls=True):
        batch_size = detr_ouput['pred_boxes'].shape[0]
        device = detr_ouput['pred_boxes'].device
        padding_box = torch.tensor([0.5,0.5,1,1],device=device)
        if stage in ['train','eval']:      # use groundtruth boxes during training but predicted boxes during inference
            boxes = [target['boxes'] for target in targets]
            length_label = [box.shape[0] for box in boxes]
            max_length = max(length_label)
            boxes_padded = padding_box.repeat(batch_size,2*max_length+1,1)
            for b in range(batch_size):
                box = boxes[b]
                boxes_padded[b,:len(box),:] = box
                repeat_times = (2*max_length+len(box)-1)//len(box)
                anchor_boxes = [get_noised_boxes(scale_n_l=0.3,scale_n_u=0.7,positive_negative_ratio=0.3,ground_truth=box,noise_percentage=1)for _ in range(repeat_times)]
                boxes_padded[b,len(box):-1,:] = torch.cat(anchor_boxes,dim=0)[:2*max_length-len(box)]
            attention_mask = get_attention_mask(length_label,self.num_head,device=device,boxes_mode=4)
            if not global_cls:
                attention_mask = attention_mask[:,:,:-1,:-1]   # remove the padding token
                boxes_padded = boxes_padded[:,:-1,:]
        elif stage in ['test','inference']:
            boxes = detr_ouput['pred_boxes'].detach().clone()
            instance_labels = detr_ouput['pred_logits'].sigmoid()>threshold
            length_label = [len(torch.where(label)[0]) for label in instance_labels]
            max_length = max(length_label)
            boxes_padded = padding_box.repeat(batch_size,max_length+1,1) if global_cls else padding_box.repeat(batch_size,max_length,1)
            for b in range(batch_size):
                box = boxes[b][instance_labels[b].squeeze(-1)]
                boxes_padded[b,:len(box),:] = box
                attention_mask = get_attention_mask(length_label,self.num_head,device=device,boxes_mode=4,cls_num=self.cls_num)
        return boxes_padded,length_label,attention_mask
def get_attention_mask(length_label,num_head,device='cuda',boxes_mode=0,cls_num=1):
    batch_size = len(length_label)
    max_length = max(length_label)
    if boxes_mode==1 or boxes_mode==2:  # including inference, normal, direct noise
        attention_mask = torch.ones((batch_size,num_head,max_length+cls_num,max_length+cls_num),dtype=torch.bool,device=device) if boxes_mode != 3 else torch.ones((batch_size,num_head,3*max_length+1,max_length+1),dtype=torch.bool,device=device)
        for b in range(batch_size):
            attention_mask[b,:,:,:length_label[b]+cls_num] = False
        attention_mask = attention_mask.view(batch_size*num_head,max_length+cls_num,max_length+cls_num)
    elif boxes_mode==3:
        attention_mask = torch.ones((batch_size,num_head,3*max_length+cls_num,3*max_length+cls_num),dtype=torch.bool,device=device)
        for b in range(batch_size):
            l = length_label[b]
            # instance to instance  
            attention_mask[b,:, :l, :l] = False
            # instance to cls
            attention_mask[b,:, :l, 3*l:3*l+cls_num] = False
            # cls to instance and itself
            attention_mask[b,:, 3*l:, :l] = False
            # cls to instance and itself
            attention_mask[b,:, 3*l:, :l] = False
            attention_mask[b,:, 3*l:, 3*l:] = False
            # positive to positive
            attention_mask[b,:,l:2*l, l:2*l] = False
            # negative to negative
            attention_mask[b,:, 2*l:3*l, 2*l:3*l] = False
        attention_mask = attention_mask.view(batch_size*num_head, 3*max_length+cls_num, 3*max_length+cls_num)
    elif boxes_mode==4:
        attention_mask = torch.ones((batch_size,num_head,2*max_length+cls_num,2*max_length+cls_num),dtype=torch.bool,device=device)
        for b in range(batch_size):
            length_instance = length_label[b]
            attention_mask[b,:,:length_instance,:length_instance] = False # instance to instance
            attention_mask[b,:,length_instance:-cls_num,:length_instance] = False # anchor to instance
            # anchor_idx = torch.arange(length_instance, 2*max_length, device=device)# anchor to itself
            # attention_mask[b,:,anchor_idx,anchor_idx] = False
            attention_mask[b,:,-cls_num:,:length_instance] = False
            attention_mask[b,:,-cls_num:,-cls_num:] = False
        attention_mask = attention_mask.view(batch_size*num_head,2*max_length+cls_num,2*max_length+cls_num)
    return attention_mask
def get_padding_mask(length_label,device='cuda',contrastive=True,cls_num=1):
    batch_size = len(length_label)
    max_length = max(length_label)
    if not contrastive:
        key_padding_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
        for b, length in enumerate(length_label):
            key_padding_mask[b, length+cls_num:] = True 
    else:
        key_padding_mask = torch.zeros(batch_size, 2*max_length+cls_num, dtype=torch.bool, device=device)
        for b, length in enumerate(length_label):
            key_padding_mask[b, length:2*length+cls_num] = True 
    return key_padding_mask

from util.box_ops import (box_cxcywh_to_xyxy,box_xyxy_to_cxcywh)

def get_noised_boxes(scale_n_l,scale_n_u,positive_negative_ratio,ground_truth,noise_percentage):
    '''for training, add noise to groundtruth boxes
    scale_n_l: lower bound of noise scale factor
    scale_n_u: upper bound of noise scale factor
    positive_negative_ratio: ratio of positive noise to negative noise
    ground_truth: groundtruth boxes
    noise_percentage: percentage of boxes to add noise'''
    ground_truth_ = ground_truth.clone()
    gt_mask = torch.rand(ground_truth.shape[:-1])<(1-noise_percentage)
    gt_mask = gt_mask.to(ground_truth.device)
    w_h,x_y = (ground_truth[...,2:],ground_truth[...,:2])
    pos_dist = torch.distributions.Uniform(low=scale_n_l, high=scale_n_u)
    neg_dist = torch.distributions.Uniform(low=-scale_n_u, high=-scale_n_l)
    bernoulli_dist = torch.distributions.Bernoulli(probs=positive_negative_ratio)
    mask_w_h = bernoulli_dist.sample(w_h.shape).bool().to(w_h.device)
    mask_x_y = torch.bernoulli(torch.full_like(x_y,0.5)).bool().to(x_y.device)  ## for control the posibility of positive values and negative values
    scale_factors_wh = torch.where(mask_w_h, pos_dist.sample(w_h.shape).to(w_h.device), neg_dist.sample(w_h.shape).to(w_h.device))
    w_h_noise = scale_factors_wh * w_h
    scale_factors_xy = torch.where(mask_x_y, pos_dist.sample(x_y.shape).to(x_y.device), neg_dist.sample(x_y.shape).to(x_y.device))
    x_y_noise = scale_factors_xy * w_h*0.5
    box_noise = torch.cat([x_y_noise,w_h_noise],dim=-1)
    box_noise = ground_truth+box_noise
    box_noise[gt_mask] = ground_truth_[gt_mask]
    box_noise = torch.clamp(box_cxcywh_to_xyxy(box_noise),min=0,max=1) ## constrain the box to be in [0,1], but the 
    return box_xyxy_to_cxcywh(box_noise)

class initial_decoder_layer(DeformableTransformerDecoderLayer):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,restrict_attn = True):
        super(initial_decoder_layer,self).__init__(d_model, d_ffn, dropout, activation, n_levels, n_heads, n_points,restrict_attn)
    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                                   reference_points,
                                   src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt+self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt
    

class decouple_decoder(nn.Module):
    def __init__(self,initial_layer,decoder_layer,num_layers):
        super().__init__()
        self.inital_layer = initial_layer
        self.layers = _get_clones(decoder_layer, num_layers-1)
    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None,attention_mask=None):
        if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
        else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
        tgt = self.inital_layer(tgt, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)
        for layer in self.layers:
            tgt = layer(tgt, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask,attn_mask=attention_mask)
        return tgt


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, method='topk'):
        super().__init__()
        if method is None:
            method = 'topk'
        assert method in ['topk', 'label_topk', 'label']
        self.method = method
        self.fn = dict(topk=postprocess_topk,
                       label_topk=postprocess_label_topk,
                       label=postprocess_label)[method]

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
       
        boxes, labels, scores = self.fn(outputs)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results
    
def postprocess_topk(outputs):
    """
    This is the default procedure for postprocessing the output of DETR.
    We have modified it so that k is one third of the number of queries, rather than 100.
    """
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']
    prob = out_logits.sigmoid()
    k    = int(out_logits.size(-2) // 3)
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), k, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
    return boxes, labels, scores

def postprocess_label_topk(outputs):
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']
    prob = out_logits.sigmoid()
    
    # get score and label of each query
    scores, labels  = prob.max(-1)
    # select top-k queries
    k  = int(out_logits.size(-2) // 3)
    topk_scores, topk_indices = torch.topk(scores, k, dim=1)
    
    # gather
    labels = torch.gather(labels, 1, topk_indices)
    scores = torch.gather(scores, 1, topk_indices)
    boxes  = torch.gather(out_bbox, 1, topk_indices.view(-1,k,1).repeat(1,1,4))
    
    # convert format
    boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    return boxes, labels, scores

def postprocess_label(outputs):
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']
    prob = out_logits.sigmoid()
    
    # get score and label of each query
    scores, labels  = prob.max(-1)
    
    # convert format
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    return boxes, labels, scores