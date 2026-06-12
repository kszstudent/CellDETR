from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
from .ms_deform_attn import MSDeformAttn
 ## restricting the attention range into the boxes of queries
class RestrictedMSDeformAttn(MSDeformAttn):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__(d_model, n_levels, n_heads, n_points)
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        assert reference_points.shape[-1] == 4, 'reference_points.shape[-1] must be 4'
        sampling_offsets = F.tanh(sampling_offsets)   ## to restrict the attention range into the boxes of queries
        sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets * reference_points[:, :, None, :, None, 2:] * 0.5
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output