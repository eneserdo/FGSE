"""
Notes:

Shape abbreviations:
    T: temporal dimension
    N: batch dimension
    F: feature dimension
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch_geometric.nn as g_nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_edge

from dataset import ACTIONS, OBJECTS, RELATIONS
from utils import (align_embeddings_to_right, get_filters_for_imbalance,
                   print_warning)


# Borrowed from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0, max_len=5000, use_library=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding
    """

    def __init__(self, seg_len, d_model, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.nn.Parameter(torch.zeros(seg_len, 1, d_model), requires_grad=True)
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


class GraphModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.filter_it = args.filter_it
        self.use_hand_pooling = args.use_hand_pooling
        self.edge_dropout = args.edge_dropout
        self.use_embedding = args.use_embedding
        self.use_pos = args.use_pos

        self.use_embedding_edge = args.use_embedding_edge
        self.use_pos_edge = args.use_pos_edge

        self.return_attention = args.return_attention
        self.dist_threshold = args.dist_threshold

        if args.use_embedding:
            self.embedding = torch.nn.Embedding(len(OBJECTS), len(OBJECTS)) # CHECKME: Second arg might be different than in_channels

        if args.use_embedding_edge:
            # not reasonable
            # raise NotImplementedError("use_embedding_edge is not implemented yet")
            self.edge_embedding = torch.nn.Linear(args.num_edge_feat, args.num_edge_feat)
                    

        assert args.norm_layer != "none", "norm_layer should not be none"
        assert args.norm_layer != "PairNorm", "PairNorm is not implemented yet"
        # assert args.num_layers >= 2, "num_layers should be at least 2"

        NormLayerType = getattr(g_nn, args.norm_layer)

        if args.norm_layer == "BatchNorm":
            bn_transfer_string = "x -> x"
        else:
            bn_transfer_string = "x, batch -> x"

        
        
        if args.return_attention:
            if args.gnn_type in ["TransformerConv", "GATv2Conv"]:
                # extra_str = ", size, flag"
                layer_str = "x, edge_index, edge_attr, flag -> x, (edge_index, attention_weights{})"
                module_str = "x, edge_index, edge_attr, batch, flag"

            elif args.gnn_type == "GATConv":
                raise NotImplementedError(f"return_attention is not implemented for {args.gnn_type}")
                layer_str = "x, edge_index, edge_attr, size, flag -> x, (edge_index, attention_weights{})"
                module_str = "x, edge_index, edge_attr, batch, size, flag"

            else:
                raise NotImplementedError(f"return_attention is not implemented for {args.gnn_type}")
        else:
            layer_str = "x, edge_index, edge_attr -> x"
            module_str = "x, edge_index, edge_attr, batch"

        module_list = []

        # All layers
        assert args.num_layers == len(args.channels) == len(args.num_heads), f"{args.num_layers=}, {len(args.channels)=}, {len(args.num_heads)=}"
        prev_ch = args.in_channels
        for layer_idx in range(args.num_layers):
            layer_ch, layer_head = args.channels[layer_idx], args.num_heads[layer_idx]
            
            next_ch = layer_ch * layer_head if not args.no_concat else layer_ch
            
            module_list.extend([(getattr(g_nn, args.gnn_type)(prev_ch, layer_ch, heads=layer_head, edge_dim=args.num_edge_feat, dropout=args.dropout, concat=not args.no_concat), layer_str.format(layer_idx)), 
                                (nn.SELU(), "x -> x"), 
                                (NormLayerType(next_ch), bn_transfer_string)])
            prev_ch = next_ch

        self.gnn = g_nn.Sequential(module_str, module_list)

    def _hands_pooling(self, x, embeddings):
        
        labels = torch.argmax(x, dim=1)

        # Below code is much more optimal, use when data redundancy is fixed.
        l_emb, r_emb = embeddings[labels == OBJECTS.index("LeftHand")], embeddings[labels == OBJECTS.index("RightHand")]
        assert l_emb.shape == r_emb.shape

        return l_emb, r_emb
   
    def forward(self, dataBatch: Batch):
        
        
        x, edge_attr, edge_index, batch = dataBatch.x, dataBatch.edge_attr, dataBatch.edge_index, dataBatch.batch

        # filter distant edges
        if self.dist_threshold is not None:
            edge_mask_dist = torch.norm(dataBatch.dists, dim=1) < self.dist_threshold
        
            edge_attr = edge_attr[edge_mask_dist]
            edge_index = edge_index[:, edge_mask_dist]

        if hasattr(dataBatch, "pos"):
            pos_3d = dataBatch.pos.to(torch.float32) # somehow it is float64

        if self.edge_dropout != 0.0:
            edge_index, edge_mask = dropout_edge(edge_index, p=self.edge_dropout)
            edge_attr = edge_attr[edge_mask]

        if self.use_embedding:
            x_ = self.embedding(x.argmax(dim=1).long())
        else:
            x_ = x
        
        if self.use_pos:
            x_ = torch.cat([x_, pos_3d], dim=-1)

        if self.use_embedding_edge:
            # array -> bits -> int -> embedding
            # bit_mask = torch.tensor([2**i for i in range(edge_attr.shape[-1])], device=edge_attr.device).to(torch.float32)
            # edge_attr = self.edge_embedding((edge_attr @ bit_mask).long())
            edge_attr = self.edge_embedding(edge_attr)
        
        if self.use_pos_edge:
            edge_attr = torch.cat([edge_attr, pos_3d[edge_index[0]] - pos_3d[edge_index[1]]], dim=-1)


        if self.return_attention:
            H = self.gnn(x_, edge_index, edge_attr, batch, True)
        else:
            H = self.gnn(x_, edge_index, edge_attr, batch)

        if self.use_hand_pooling:
            left_embeddings, right_embeddings = self._hands_pooling(x, H)
        else:
            # A bit inefficient implementation
            graph_embedding = g_nn.pool.global_mean_pool(H, batch)
            left_embeddings = graph_embedding
            right_embeddings = graph_embedding
        
        return left_embeddings, right_embeddings



class ActionRecognition(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.filter_it = args.filter_it
        self.use_hand_pooling = args.use_hand_pooling
        self.edge_dropout = args.edge_dropout
        self.temporal_type = args.temporal_type
        self.merged_pred = args.merged_pred
        self.use_embedding = args.use_embedding
        self.return_attention = args.return_attention
        self.causal = args.causal
        
        self.gnn = GraphModule(args)
        
        gnn_output_ch = args.channels[-1] * args.num_heads[-1] if not args.no_concat else args.channels[-1]

        if self.merged_pred == "early":
            gnn_output_ch *= 2

        if args.dim_feedforward == 0:
            # This is was the previous default value, so I am keeping it
            # But this makes the embedding size > dim_feedforward, which is not good
            args.dim_feedforward = args.lstm_hidden_size // 4 * 3

        if args.temporal_type in ["lstm", "bi"]:
            if args.num_of_temp_layers != 1:
                raise NotImplementedError("args.num_of_temp_layers is not 1 as used to be. Are you sure?")
            num_rnn_layers = 1  # Lets make it constant
            is_bidirectional = ("bi" == args.temporal_type)

            # To have approx same number of parameters
            lstm_hidden_size = args.lstm_hidden_size * 2 // 3 if is_bidirectional else args.lstm_hidden_size

            self.temporal_layer = nn.LSTM(gnn_output_ch, lstm_hidden_size, num_rnn_layers,
                                       bidirectional=is_bidirectional, dropout=args.dropout)
            lstm_hidden_size = lstm_hidden_size * 2 if is_bidirectional else lstm_hidden_size

        elif args.temporal_type == "tr":
            # Default dropout is 0.1

            self.temporal_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(gnn_output_ch, args.num_heads_tr, args.dim_feedforward, norm_first=args.norm_first, dropout=args.dropout_tr), args.num_of_temp_layers)
            self.PE_coef = math.sqrt(gnn_output_ch)

            if args.pos_enc == "original":
                # if merged_pred == "attention":
                #     self.pos_encoder = PositionalEncodingPseudo2D(effective_out_ch2)
                # else:
                #     self.pos_encoder = PositionalEncodingPseudo2D(effective_out_ch2)
                self.pos_encoder = PositionalEncoding(gnn_output_ch)
                # self.pos_encoder = PositionalEncoding1D(effective_out_ch2)

            elif args.pos_enc == "learnable":
                if args.merged_pred == "attention":
                    self.pos_encoder = LearnablePositionalEncoding(args.temporal_length*2, gnn_output_ch)
                else:    
                    self.pos_encoder = LearnablePositionalEncoding(args.temporal_length, gnn_output_ch)
                
                # raise NotImplementedError("Learnable positional encoding is not implemented yet")
            
            elif args.pos_enc == "none":
                self.pos_encoder = lambda x: x
            
            else:
                raise NotImplementedError(f"pos_enc: {args.pos_enc} is not implemented")
            
            # self.cls_tokens = torch.nn.Parameter(torch.randn(1, 1, effective_out_ch2)).expand(self.batch_size, -1, -1)

            lstm_hidden_size = gnn_output_ch


        elif args.temporal_type == "none":
            # self.temporal_layer = torch.nn.Sequential(torch.nn.Linear(effective_out_ch2, effective_out_ch2), 
            #                                     torch.nn.SELU(),
            #                                     torch.nn.Linear(effective_out_ch2, num_action_classes))
            self.temporal_layer = torch.nn.Linear(gnn_output_ch, args.num_action_classes)

        def build_mlp(in_feat,out_feat, hid_feat=None):
            # hid_feat = (in_feat+out_feat)//2 if hid_feat == None else hid_feat
            # return Sequential(nn.Linear(in_feat, hid_feat), nn.SELU(), nn.Linear(hid_feat, out_feat))
            return nn.Linear(in_feat,out_feat)

        if self.temporal_type != "none":
            if self.merged_pred == "late":
                self.left_mlp = build_mlp(lstm_hidden_size*2, args.num_action_classes)
                self.right_mlp = build_mlp(lstm_hidden_size*2, args.num_action_classes)

            elif self.merged_pred == "early":
                # self.left_mlp = build_mlp(lstm_hidden_size, args.num_action_classes)
                # self.right_mlp = build_mlp(lstm_hidden_size, args.num_action_classes)
                self.mlp = build_mlp(lstm_hidden_size//2, args.num_action_classes)

            elif self.merged_pred in ["none", "attention"]:    
                self.mlp = build_mlp(lstm_hidden_size, args.num_action_classes)


    def forward(self, dataBatchList: List[Batch]):

        left_embedding_list = []
        right_embedding_list = []

        #### Graph part ####
        # processes each frame time-independently
        for databatch in dataBatchList:  

            left_embeddings, right_embeddings = self.gnn(databatch)
            left_embedding_list.append(left_embeddings)
            right_embedding_list.append(right_embeddings)


        left_sequencial_embeddings = torch.stack(left_embedding_list) # Shape: (T, N, F)
        right_sequencial_embeddings = torch.stack(right_embedding_list) # Shape: (T, N, F)

        if self.temporal_type != "none":
            # Filtering imbalanced data only during training
            if self.filter_it and self.training:
                raise NotImplementedError("Filtering may be broken, please check it")
                # get filters
                left_filter, right_filter = get_filters_for_imbalance(dataBatchList)

                # filter it
                right_sequencial_embeddings[~right_filter] = 0
                left_sequencial_embeddings[~left_filter] = 0

                # align non-zero embeddings to the right
                # TODO: Insert assert statement to make sure it is working as expectedly            
                right_sequencial_embeddings = align_embeddings_to_right(right_sequencial_embeddings, right_filter)
                left_sequencial_embeddings = align_embeddings_to_right(left_sequencial_embeddings, left_filter)

            if self.causal:
                causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(left_sequencial_embeddings.shape[0])
            else:
                causal_mask = None

            # Enabling the information flow between hands
            if self.merged_pred == "early": # concat in feature dim
                both_sequencial_embeddings = torch.concat([left_sequencial_embeddings, right_sequencial_embeddings], dim=-1)
                # shape = (T, N, F*2)

            elif self.merged_pred == "attention": # concat in time dim
                both_sequencial_embeddings = torch.concat([left_sequencial_embeddings, right_sequencial_embeddings], dim=0)
                # shape = (T*2, N, F)
                causal_mask = causal_mask.repeat(2, 2) if causal_mask is not None else None

            elif self.merged_pred in ["late", "none"]: # concat in batch dim
                both_sequencial_embeddings = torch.concat([left_sequencial_embeddings, right_sequencial_embeddings], dim=1) 
                # shape = (T, N*2, F)
            else:
                raise NotImplementedError(f"merged_pred: {self.merged_pred} is not implemented")
            

            #### Temporal part ####
            if self.temporal_type == "tr":
                both_sequencial_embeddings = self.pos_encoder(both_sequencial_embeddings * self.PE_coef)
                both_out = self.temporal_layer(both_sequencial_embeddings, causal_mask)

            elif self.temporal_type == "bi":
                both_out, _ = self.temporal_layer(both_sequencial_embeddings)

            elif self.temporal_type == "lstm":
                both_out, _ = self.temporal_layer(both_sequencial_embeddings)
            
            elif self.temporal_type == "edtr":
                # Encoder-Decoder Transformer, mask the decoder part 
                both_out = self.temporal_layer(both_sequencial_embeddings, both_sequencial_embeddings, None, self.decoder_mask)
                
            #### Prediction part ####
            if self.merged_pred == "late":
                # merged_out_s = self._merge_hands(out_s)
                # action_prediction = torch.cat((self.left_mlp(merged_out_s), self.right_mlp(merged_out_s)),dim=0)
                # both_out.shape = (T, N*2, F)
                # both_out_ml.shape = (T, N, F*2)
                
                both_out_left, both_out_right = torch.tensor_split(both_out, 2, dim=1) 

                both_out_ml = torch.cat([both_out_left, both_out_right], dim=-1)
                
                action_prediction = torch.cat((self.left_mlp(both_out_ml), self.right_mlp(both_out_ml)), dim=1)
                

            elif self.merged_pred == "early":
                # both_out.shape = (T, N, F*2)
                
                # action_prediction = torch.cat((self.left_mlp(both_out), self.right_mlp(both_out)), dim=1)
                
                # Old way
                both_out_left, both_out_right = torch.tensor_split(both_out, 2, dim=2)
                # # single mlp
                both_out_ml = torch.cat([both_out_left, both_out_right], dim=1)
                action_prediction = self.mlp(both_out_ml)

                # # separate mlps
                # # action_prediction = torch.cat((self.left_mlp(both_out_left), self.right_mlp(both_out_right)), dim=1)

            elif self.merged_pred == "attention":

                # both_out.shape = (T*2, N, F)

                both_out_left, both_out_right = torch.tensor_split(both_out, 2, dim=0)

                both_out = torch.cat([both_out_left, both_out_right], dim=1)

                action_prediction = self.mlp(both_out)

            elif self.merged_pred == "none":
                # both_out.shape = (T, N*2, F)
                action_prediction = self.mlp(both_out)
                
            else:
                raise NotImplementedError(f"merged_pred: {self.merged_pred} is not implemented")

        else:
            action_prediction_left = self.temporal_layer(left_sequencial_embeddings)
            action_prediction_right = self.temporal_layer(right_sequencial_embeddings)
            action_prediction = torch.cat((action_prediction_left, action_prediction_right), dim=1) 

        return action_prediction


class ActionRecognitionMonograph(nn.Module):
    def __init__(self, args):
        """ Accepts kwargs for non-general params of gnns """
        super().__init__()

        self.filter_it = args.filter_it
        self.edge_dropout = args.edge_dropout
        self.use_embedding = args.use_embedding
        self.num_actions = args.num_action_classes
        self.temporal_length = args.temporal_length
        self.return_attention = args.return_attention

        self.gnn = GraphModule(args)


        gnn_output_ch = args.out_channels * args.num_heads

        self.head = nn.Sequential(nn.Linear(gnn_output_ch, gnn_output_ch),
                                  nn.SELU(),
                                  nn.Linear(gnn_output_ch, args.num_action_classes))



    def forward(self, databatch: Batch):

        bs = databatch.batch[-1] + 1
        left_embeddings, right_embeddings = self.gnn(databatch) 
        # shape: (n*b, f)

        hands_embeddings = torch.cat((left_embeddings, right_embeddings), dim=0)
        
        action_prediction = self.head(hands_embeddings)

        left_actions, right_actions = torch.tensor_split(action_prediction, 2, dim=0)

        left_actions = left_actions.reshape(bs, self.temporal_length, self.num_actions).permute(1, 0, 2)
        right_actions = right_actions.reshape(bs, self.temporal_length, self.num_actions).permute(1, 0, 2)

        return torch.cat((left_actions, right_actions), dim=1)



def make_model(args):
    if args.use_pos:
        in_channels = len(OBJECTS) + 3
    elif args.use_vf:
        in_channels = 256
    else:
        in_channels = len(OBJECTS)

    if args.use_pos_edge:
        num_edge_feat = len(RELATIONS) + 3
    else:
        num_edge_feat = len(RELATIONS)
    
    if args.monograph:
        args.num_action_classes = len(ACTIONS)
        args.use_hand_pooling = not args.use_global_pooling
        args.filter_it = args.filter_idle_hold
        args.num_edge_feat = num_edge_feat + 1
        args.in_channels = in_channels

        model = ActionRecognitionMonograph(args)
    else:
        args.num_action_classes = len(ACTIONS)
        args.use_hand_pooling = not args.use_global_pooling
        args.filter_it = args.filter_idle_hold
        args.num_edge_feat = num_edge_feat
        args.in_channels = in_channels

        model = ActionRecognition(args)

    return model