import torch
from .decouple_detr import build_decouple_decoder
from .decouple_detr.decouple_detr import (Decouple_DeformableDETR,PostProcess)
from .decouple_detr.criterions import Decouple_SetCriterion
from .backbone import build_backbone
from .deformable_detr.deformable_transformer import build_deforamble_transformer
from .deformable_detr.matcher import build_matcher
from .deformable_detr.segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,)
from .deformable_detr.deformable_detr import build_detr
def build_model(args):
    num_classes = args.model.num_classes
    device = torch.device(args.experiment.device)
    decouple_decoder = build_decouple_decoder(args)
    detr = build_detr(args.model)
    model = Decouple_DeformableDETR(
            detr,
            decouple_decoder=decouple_decoder,
            decouple_classes=args.model.num_classes,
            training_type=args.model.training_type,
            boxes_mode=args.model.boxes_mode,
        )
    if args.model.masks:
        model = DETRsegm(model, freeze_detr=(args.loss.frozen_weights is not None))
    matcher = build_matcher(args.matcher)
    weight_dict = {'loss_ce': args.loss.class_loss_coef,'loss_giou': args.loss.giou_loss_coef,'loss_bbox': args.loss.bbox_loss_coef}
    if  args.experiment.data_type == 'contrastive':
        weight_dict['loss_contrastive'] = args.loss.contrastive_loss_coef
    elif args.experiment.data_type == 'semi':
        weight_dict['loss_semi'] = args.loss.semi_loss_coef
    elif (args.experiment.data_type == 'weak'):
        weight_dict['onehot_loss'] = args.loss.onehot_loss_coef
        weight_dict['loss_weak'] = args.loss.weak_class_coef
    elif args.experiment.data_type == 'supervised':
        weight_dict['loss_label'] = args.loss.label_loss_coef

    if args.model.masks:
        weight_dict["loss_mask"] = args.loss.mask_loss_coef
        weight_dict["loss_dice"] = args.loss.dice_loss_coef
    # TODO this is a hack
    if args.model.aux_loss:
        aux_weight_dict = {}
        for i in range(args.model.transformer.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'decouple']
    if args.model.masks:
        losses += ["masks"]
    criterion = Decouple_SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,losses=losses,train_type=args.model.training_type,data_type=args.experiment.data_type,box_mode=args.model.boxes_mode)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(method='label')}
    if args.model.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.model.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors