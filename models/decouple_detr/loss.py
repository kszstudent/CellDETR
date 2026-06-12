import torch
from torch.cuda import temperature
import torch.nn.functional as F

from eval.pannuke_eval import pair_coordinates
from ..deformable_detr.segmentation import sigmoid_focal_loss
## ----supervised loss(squenced version)
def loss_squenced_supervised(outputs, targets, indices,idx, num_boxes,focal_alpha,optimize_all = True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    assert 'decouple_class_logits' in outputs
    src_logits = outputs['decouple_class_logits']
    
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2],
                                dtype=torch.int64, device=src_logits.device)
    target_classes[idx] = target_classes_o

    target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                        dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

    target_classes_onehot = target_classes_onehot[:,:,:-1]
    if optimize_all:
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=focal_alpha, gamma=2)*src_logits.shape[1]
    else:
        src_logits_ = src_logits[idx]
        target_classes_onehot_ = target_classes_onehot[idx]
        loss_ce = sigmoid_focal_loss(src_logits_, target_classes_onehot_, num_boxes, alpha=focal_alpha, gamma=2)*8 # multiply 8 for a large loss value. if not, the value will be too small.
    losses = {'loss_label': loss_ce}
    return losses

def loss_parallel_supervised(outputs, targets, num_boxes,focal_alpha):
    '''for parallel decouple training'''
    assert'instance_length' in outputs
    src_logits = outputs['decouple_class_logits']
    length_label = outputs['instance_length']
    target_classes_o = torch.cat([t["labels"] for t in targets])
    instance_logits = []
    for i,b in enumerate(length_label):
        instance_logits.append(src_logits[i, :b, :])
    instance_logits = torch.cat(instance_logits, dim=0)
    # test_sig = torch.sigmoid(instance_logits)
    target_classes_onehot = torch.zeros([instance_logits.shape[0], instance_logits.shape[1]],dtype=instance_logits.dtype, layout=instance_logits.layout, device=instance_logits.device)
    target_classes_onehot.scatter_(1, target_classes_o.unsqueeze(1), 1)
    loss_ce = sigmoid_focal_loss(instance_logits, target_classes_onehot, num_boxes, alpha=focal_alpha, gamma=2)*16
    losses = {'loss_label': loss_ce}
    return losses
## ----weak supervised loss(squenced version)
def loss_squenced_weak(outputs, targets,idx):
    """Classification loss -weak supervised data(squenced version)
    we use softmax to normalize the weak logits,
    and kl divergence to calculate the loss.
    """
    assert 'decouple_class_logits' in outputs
    src_logits = outputs['pred_logits']
    weak_logits = outputs['decouple_class_logits']
    batch_size= weak_logits.shape[0]
    front_logits = weak_logits[idx] ## should don't contain background.
    front_pred = F.softmax(front_logits,dim=-1,) 
    sum_tensor = torch.zeros((batch_size,len(idx[0]))).to(src_logits.device)
    for i in range(batch_size):
        sum_tensor[i][torch.where(idx[0]==i)] = (1/len(torch.where(idx[0]==i)[0]))
    front_pred_cum = torch.mm(sum_tensor,front_pred)
    weak_labels = torch.cat([t['weak_label'].unsqueeze(0) for t in targets],dim=0).float()
    loss_weak = F.kl_div(torch.log(front_pred_cum),weak_labels,reduction='mean')*weak_labels.shape[-1]
    loss_entropy = -torch.sum(front_pred * torch.log(front_pred + 1e-8), dim=1).mean()
    losses = {}
    losses['loss_weak'] = loss_weak
    losses['onehot_loss'] = loss_entropy
    return losses
    
def loss_parallel_weak(outputs,targets):
    '''for weak data using parallel training '''
    assert'instance_length' in outputs
    src_logits = outputs['decouple_class_logits']
    length_label = outputs['instance_length']
    src_dis = F.softmax(src_logits,dim=-1)
    weak_dis = []
    instance_dis = []
    for i,b in enumerate(length_label):
        weak_dis.append((src_dis[i, :b, :].sum(dim=0))/b)
        instance_dis.append(src_dis[i, :b, :])
    weak_dis = torch.stack(weak_dis,dim=0)
    instance_dis = torch.cat(instance_dis,dim=0)
    weak_labels = torch.cat([t['weak_label'].unsqueeze(0) for t in targets],dim=0).float()
    loss_weak = F.kl_div(torch.log(weak_dis),weak_labels,reduction='mean')*weak_labels.shape[-1]
    loss_entropy = -torch.sum(instance_dis * torch.log(instance_dis + 1e-8), dim=1).mean()
    losses = {}
    losses['loss_weak'] = loss_weak
    losses['onehot_loss'] = loss_entropy
    return losses
## -----semi supervised loss(squenced version)
def loss_squeenced_semi(outputs, targets,indices,idx):
    '''use semi-supervised data to train or finetune the model(squenced version)'''
    assert 'decouple_class_logits' in outputs
    src_logits = outputs['decouple_class_logits']
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    labeled_target_class = target_classes_o[torch.where(target_classes_o!=-1)].to(torch.int64)
    src_logits_ = src_logits[idx][torch.where(target_classes_o!=-1)]
    target_classes_onehot = torch.zeros([src_logits_.shape[0], src_logits_.shape[1] + 1],
                                        dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    target_classes_onehot.scatter_(1, labeled_target_class.unsqueeze(-1), 1)

    target_classes_onehot = target_classes_onehot[:,:-1]
    loss_ce = sigmoid_focal_loss(src_logits_, target_classes_onehot, len(src_logits_), alpha=focal_alpha, gamma=2)*8
    losses = {'loss_semi': loss_ce}
    return losses 

def loss_parallel_semi(outputs, targets,focal_alpha):
    '''use semi-supervised data to train or finetune the model(parallel version)'''
    assert 'decouple_class_logits' in outputs
    src_logits = outputs['decouple_class_logits']
    length_label = outputs['instance_length']
    target_classes_o = torch.cat([t["labels"] for t in targets])
    labeled_target_class = target_classes_o[torch.where(target_classes_o!=-1)].to(torch.int64)   # the global token was padding in the last position
    src_logits_ = []
    for i, l in enumerate(length_label):
        src_logits_.append(src_logits[i, :l, :])
    src_logits_ = torch.cat(src_logits_, dim=0)
    src_logits_ = src_logits_[torch.where(target_classes_o!=-1)]   # sampling  cells with lables
    target_classes_onehot = torch.zeros([src_logits_.shape[0], src_logits_.shape[1] + 1],
                                        dtype=src_logits.dtype, layout=src_logits_.layout, device=src_logits.device)
    target_classes_onehot.scatter_(1, labeled_target_class.unsqueeze(-1), 1)

    target_classes_onehot = target_classes_onehot[:,:-1]
    loss_ce = sigmoid_focal_loss(src_logits_, target_classes_onehot, len(src_logits_), alpha=focal_alpha, gamma=2)*8
    losses = {'loss_semi': loss_ce}
    return losses
## dion stategy supervised train(only parallel  version)
def loss_dino(outputs,targets,cls_num=1,focal_alpha=0.25):
    '''use dino strategy to train the model(parallel version).we conpute the focal loss for groundtruth, positive sample and negtive sample
    cls_num: the number of cls token/box
    '''
    assert'instance_length' in outputs
    src_logits = outputs['decouple_class_logits']
    length_label = outputs['instance_length']
    target_classes_o = torch.cat([t["labels"] for t in targets])
    instance_logits = []
    positive_logits = []
    negtive_logits = []
    for i,l in enumerate(length_label):
        instance_logits.append(src_logits[i, :l, :])
        positive_logits.append(src_logits[i, l:2*l, :])
        negtive_logits.append(src_logits[i, 2*l:3*l,:])
    instance_logits = torch.cat(instance_logits, dim=0)
    positive_logits = torch.cat(positive_logits, dim=0)
    negtive_logits = torch.cat(negtive_logits, dim=0)
    num_boxes = instance_logits.shape[0]
    # test_sig = torch.sigmoid(instance_logits)
    target_classes_onehot = torch.zeros([instance_logits.shape[0], instance_logits.shape[1]],dtype=instance_logits.dtype, layout=instance_logits.layout, device=instance_logits.device)
    target_classes_onehot.scatter_(1, target_classes_o.unsqueeze(1), 1)
    loss_instance = sigmoid_focal_loss(instance_logits, target_classes_onehot, num_boxes, alpha=focal_alpha, gamma=2)*16
    loss_positive = sigmoid_focal_loss(positive_logits, target_classes_onehot, num_boxes, alpha=focal_alpha, gamma=2)*16
    loss_negtive = sigmoid_focal_loss(negtive_logits,torch.zeros_like(negtive_logits), num_boxes, alpha=focal_alpha, gamma=2)*16
    losses = {'loss_label': loss_instance+0.5*loss_positive+0.5*loss_negtive}
    return losses
## ----contrastive loss(only parallel version)
def loss_contrastive(outputs,temperature=0.5):
    assert 'contrastive_logits' in outputs
    assert 'instance_length' in outputs
    contrastive_logits = outputs['contrastive_logits']
    instance_length = outputs['instance_length']
    max_length = max(instance_length)
    loss_batch = []
    for i, l in enumerate(instance_length):
        if l > 0:
            pair_index = (torch.arange(2*max_length-l))%l
            instance_patch = contrastive_logits[i, :l, :]
            anchor_patch = contrastive_logits[i, l:-1, :]
            anchor_instance_exp = torch.exp(F.cosine_similarity(anchor_patch.unsqueeze(1), instance_patch.unsqueeze(0), dim=-1)/temperature)
            postive_exp = anchor_instance_exp[torch.arange(2*max_length-l),pair_index]
            patch_loss = -torch.log(postive_exp/(anchor_instance_exp.sum(dim=-1)))
            loss_batch.append(patch_loss.mean())
    losses = {'loss_contrastive':torch.stack(loss_batch).mean()}
    return losses