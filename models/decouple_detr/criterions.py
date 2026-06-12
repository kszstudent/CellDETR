from ..deformable_detr.deformable_detr import SetCriterion
import torch
import torch.nn.functional as F
from util.misc import (NestedTensor,accuracy, get_world_size,is_dist_avail_and_initialized,)
import copy
from ..deformable_detr.segmentation import sigmoid_focal_loss

from .loss import (loss_dino, loss_parallel_weak,loss_squenced_weak,loss_parallel_supervised,loss_squenced_supervised,loss_parallel_semi,loss_contrastive)
class Decouple_SetCriterion(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, losses, train_type='parallel',data_type='supervised',box_mode=0,focal_alpha=0.25):
        super().__init__(num_classes, matcher, weight_dict, losses,focal_alpha)
        assert train_type in ['parallel','squenced','contrastive'],'train_type must be parallel or squenced or contrastive'
        self.train_type = train_type
        assert data_type in ['supervised','weak','semi','contrastive'],'data_type must be supervised or weak or semi or contrastive'
        self.data_type = data_type
        assert box_mode in [0,1,2,3]
        self.box_mode = box_mode
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2],src_logits.shape[-1],
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = 0

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        return losses
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.sigmoid() > 0.4).sum(1)
        card_err = F.l1_loss(card_pred.squeeze(-1).float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_squenced(self, outputs, targets, indices, num_boxes, log=True):
        """Caculate the loss for squenced decouple training
        """
        idx = self._get_src_permutation_idx(indices)
        if self.data_type == 'supervised':
            losses = loss_squenced_supervised(outputs, targets, indices,idx, num_boxes,focal_alpha=self.focal_alpha,optimize_all=False)
        elif self.data_type == 'weak':
            losses = loss_squenced_weak(outputs, targets,idx,)
        return losses
    def loss_parallel(self, outputs, targets, indices, num_boxes, log=True):
        '''for parallel decouple training'''
        # idx = self._get_src_permutation_idx(indices)
        if self.data_type == 'supervised':
            if self.box_mode == 3:
                losses = loss_dino(outputs, targets,focal_alpha=self.focal_alpha)
            else:
                losses = loss_parallel_supervised(outputs, targets,num_boxes,focal_alpha=self.focal_alpha)
        elif self.data_type == 'weak':
            losses = loss_parallel_weak(outputs, targets,)
        elif self.data_type == 'semi':
            losses = loss_parallel_semi(outputs, targets,focal_alpha=self.focal_alpha)
        return losses
    def loss_contrastive(self, outputs, targets, indices, num_boxes, log=True):
        losses = loss_contrastive(outputs)
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        if self.train_type == 'contrastive':
            loss_map['decouple'] = self.loss_contrastive
        elif self.train_type == 'parallel':
            loss_map['decouple'] = self.loss_parallel
        elif self.train_type == 'squenced':
            loss_map['decouple'] = self.loss_squenced
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        targets_ = [{'image_id':t['image_id'],'labels':torch.zeros_like(t['labels'],dtype=torch.long),'boxes':t['boxes']}for t in targets]
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets_)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets_)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    if loss == 'decouple':
                        continue     ### we add this.
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses
