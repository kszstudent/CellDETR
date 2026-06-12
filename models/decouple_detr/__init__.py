from .decouple_detr import (initial_decoder_layer,DeformableTransformerDecoderLayer,
                             decouple_decoder )
def build_decouple_decoder(args):
    initial_layer = initial_decoder_layer(args.model.hidden_dim,args.model.transformer.dim_feedforward,
                                            args.model.transformer.dropout, args.model.transformer.activation,
                                            args.model.num_feature_levels,args.model.transformer.nheads,args.model.transformer.dec_n_points,args.model.transformer.restrict_attn)
    decoder_layer = DeformableTransformerDecoderLayer(args.model.hidden_dim,args.model.transformer.dim_feedforward,
                                            args.model.transformer.dropout, args.model.transformer.activation,
                                            args.model.num_feature_levels,args.model.transformer.nheads,args.model.transformer.dec_n_points,args.model.transformer.restrict_attn,args.experiment.data_type=='contrastive')
    decoder = decouple_decoder(initial_layer,decoder_layer,args.model.num_layer_feature_decoder)
    return decoder