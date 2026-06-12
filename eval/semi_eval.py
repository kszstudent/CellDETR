from .pannuke_eval import *
def cell_type_detection_scores_semi(
    paired_true,
    paired_pred,
    unpaired_true,
    unpaired_pred,
    type_id,
    w: List = [2, 2, 1, 1],
    exhaustive: bool = True,
):
    type_samples = (paired_true == type_id) | (paired_pred == type_id)

    paired_true = paired_true[type_samples]
    paired_pred = paired_pred[type_samples]
# fixing for ignore -1
    tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
    tn_dt = ((paired_true != type_id) & (paired_pred != type_id)&(paired_true!=-1)).sum()
    fp_dt = ((paired_true != type_id) & (paired_pred == type_id)&(paired_true!=-1)).sum()
    fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

    if not exhaustive:
        ignore = (paired_true == -1).sum()
        fp_dt -= ignore

    fp_d = (unpaired_pred == type_id).sum()  #
    fn_d = (unpaired_true == type_id).sum()

    prec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[0] * fp_dt + w[2] * fp_d)
    rec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[1] * fn_dt + w[3] * fn_d)

    f1_type = (2 * (tp_dt + tn_dt)) / (
        2 * (tp_dt + tn_dt) + w[0] * fp_dt + w[1] * fn_dt + w[2] * fp_d + w[3] * fn_d
    )
    return f1_type, prec_type, rec_type
class SemiCellDetectionMetric(CellDetectionMetric):
    def __init__(self, num_classes : int, 
                       thresholds : Union[int, List[int]], 
                       max_pair_distance : float = 12,
                       class_names : Optional[List[str]] = None,
                       *args, **kwargs):
        super().__init__(num_classes, thresholds, max_pair_distance, class_names, *args, **kwargs)
    def _compute(self, true_cents, true_labels, pred_cents, pred_labels, pred_scores):
        # metrics
        all_metrics = dict()
        # compute metrics at different thresholds
        for threshold in self.thresholds:
            # detection scores
            paired_all = []  # unique matched index pair
            unpaired_true_all = (
                []
            )  # the index must exist in `true_inst_type_all` and unique
            unpaired_pred_all = (
                []
            )  # the index must exist in `pred_inst_type_all` and unique
            true_inst_type_all = []  # each index is 1 independent data point
            pred_inst_type_all = []  # each index is 1 independent data point

            # for detections scores
            true_idx_offset = 0
            pred_idx_offset = 0

            # for each image
            for i in range(len(true_cents)):
                # get the mask accordint to the threshold
                mask = pred_scores[i] >= threshold

                # get the true and pred centroids and labels
                true_cents_i  = true_cents[i]
                true_labels_i = true_labels[i]
                pred_cents_i = pred_cents[i][mask]
                pred_labels_i = pred_labels[i][mask]

                # no predictions / no ground truth
                if true_cents_i.shape[0] == 0:
                    true_cents_i = np.array([[0, 0]])
                    true_labels_i = np.array([0])
                if pred_cents_i.shape[0] == 0:
                    pred_cents_i = np.array([[0, 0]])
                    pred_labels_i = np.array([0])

                # pairing
                paired, unpaired_true, unpaired_pred = pair_coordinates(
                    true_cents_i, pred_cents_i, self.max_pair_distance)
                
                # accumulating
                true_idx_offset = (
                    true_idx_offset + true_inst_type_all[-1].shape[0] if i != 0 else 0
                )
                pred_idx_offset = (
                    pred_idx_offset + pred_inst_type_all[-1].shape[0] if i != 0 else 0
                )
                true_inst_type_all.append(true_labels_i)
                pred_inst_type_all.append(pred_labels_i)

                # increment the pairing index statistic
                if paired.shape[0] != 0:  # ! sanity
                    paired[:, 0] += true_idx_offset
                    paired[:, 1] += pred_idx_offset
                    paired_all.append(paired)

                unpaired_true += true_idx_offset
                unpaired_pred += pred_idx_offset
                unpaired_true_all.append(unpaired_true)
                unpaired_pred_all.append(unpaired_pred)
            
            paired_all = np.concatenate(paired_all, axis=0) if len(paired_all) != 0 else np.empty((0,2), dtype=np.int64)
            unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
            unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
            true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
            pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)
            paired_true_type = true_inst_type_all[paired_all[:, 0]]
            paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
            unpaired_true_type = true_inst_type_all[unpaired_true_all]
            unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

            # compute the detection scores
            f1_d, prec_d, rec_d = cell_detection_scores(
                paired_true=paired_true_type,
                paired_pred=paired_pred_type,
                unpaired_true=unpaired_true_type,
                unpaired_pred=unpaired_pred_type)
            nuclei_metrics = {
                "detection": {
                    "f1": f1_d,
                    "prec": prec_d,
                    "rec": rec_d,
                },
            }
            
            # compute the classification scores
            if self.num_classes > 1: # if num_classes is 1, only detection scenario
                for nuc_type in range(self.num_classes):
                    f1_cell, prec_cell, rec_cell = cell_type_detection_scores_semi(
                    paired_true_type,
                    paired_pred_type,
                    unpaired_true_type,
                    unpaired_pred_type,
                    nuc_type,
                    )
                    nuclei_metrics[ self.class_names[nuc_type] ] = {
                        "f1": f1_cell,
                        "prec": prec_cell,
                        "rec": rec_cell,
                    }
            
            all_metrics["th"+str(threshold).replace(".","")] = nuclei_metrics
        # print free GPU memory
        return all_metrics