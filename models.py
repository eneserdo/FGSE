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
# from positional_encodings.torch_encodings import (PositionalEncoding1D,
#                                                   PositionalEncoding2D)
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
    
# class PositionalEncodingPseudo2D(nn.Module):
#     """
#     Same as previous but with pseudo 2D positional encoding. Instead of (T, 2), it is ([T1, T2])
#     """

#     def __init__(self, d_model, dropout=0.1):
#         super(PositionalEncodingPseudo2D, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.pe_ = PositionalEncoding2D(d_model)
#         self.is_set = False
        
#     def _set_pe(self, shape, device):
        
#         x_dim, bs, d_model = shape

#         self.x_dim = x_dim

#         y_dim = 2
#         self.zeros = torch.zeros((bs, x_dim, y_dim, d_model), dtype=torch.float32).to(device)
        
#         self.pe_obtained = self.pe_(self.zeros)

#         self.pe_y0 = self.pe_obtained[:, :, 0, :].transpose(0, 1)

#         self.pe_y1 = self.pe_obtained[:, :, 1, :].transpose(0, 1)

#         self.pe = torch.cat([self.pe_y0, self.pe_y1], dim=1)
#         self.is_set = True


#     def forward(self, x):
#         r"""Inputs of forward function
#         Args:
#             x: the sequence fed to the positional encoder model (required).
#         Shape:
#             x: [sequence length, batch size, embed dim]
#             output: [sequence length, batch size, embed dim]
#         Examples:
#             >>> output = pos_encoder(x)
#         """

#         if not self.is_set:
#             self._set_pe(x.shape, x.device)

#         assert x.shape == self.x_dim.shape

#         x = x + self.pe

#         return self.dropout(x)


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


# Borrowed from https://pytorch.org/tutorials/beginner/translation_transformer.html#seq2seq-network-using-transformer
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = torch.nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        # self.generator = nn.nn.Linear(emb_size, tgt_vocab_size)
        # self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        # self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor = None,
                tgt_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None):
        
        embedding_size = src.shape[-1]
        scale = math.sqrt(embedding_size)
        src_emb = self.positional_encoding(src*scale)
        tgt_emb = self.positional_encoding(trg*scale)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # return self.generator(outs)
        return outs

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(src), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(tgt), memory, tgt_mask)

  

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
        assert args.num_layers >= 2, "num_layers should be at least 2"

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
        r_emb = embeddings[labels == OBJECTS.index("RightHand")]
        return r_emb
   
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
            right_embeddings = self._hands_pooling(x, H)
        else:
            # A bit inefficient implementation
            graph_embedding = g_nn.pool.global_mean_pool(H, batch)
            right_embeddings = graph_embedding
        
        return right_embeddings



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

        elif args.temporal_type == "edtr":
            # Encoder-Decoder Transformer
            self.temporal_layer = Seq2SeqTransformer(args.num_of_temp_layers, args.num_of_temp_layers, gnn_output_ch, args.num_heads_tr, args.dim_feedforward, args.dropout_tr)
            self.decoder_mask = torch.nn.Transformer.generate_square_subsequent_mask(args.temporal_length)
        

        def build_mlp(in_feat,out_feat, hid_feat=None):
            # hid_feat = (in_feat+out_feat)//2 if hid_feat == None else hid_feat
            # return Sequential(nn.Linear(in_feat, hid_feat), nn.SELU(), nn.Linear(hid_feat, out_feat))
            return nn.Linear(in_feat,out_feat)

        # if self.temporal_type != "none":
        #     if self.merged_pred == "late":
        #         self.left_mlp = build_mlp(lstm_hidden_size*2, args.num_action_classes)
        #         self.right_mlp = build_mlp(lstm_hidden_size*2, args.num_action_classes)

        #     elif self.merged_pred == "early":
        #         # self.left_mlp = build_mlp(lstm_hidden_size, args.num_action_classes)
        #         # self.right_mlp = build_mlp(lstm_hidden_size, args.num_action_classes)
        #         self.mlp = build_mlp(lstm_hidden_size//2, args.num_action_classes)

        #     elif self.merged_pred in ["none", "attention"]:    
        #         self.mlp = build_mlp(lstm_hidden_size, args.num_action_classes)
        self.mlp = build_mlp(lstm_hidden_size, args.num_action_classes)

    def forward(self, dataBatchList: List[Batch]):

        right_embedding_list = []

        #### Graph part ####
        # processes each frame time-independently
        for databatch in dataBatchList:  

            right_embeddings = self.gnn(databatch)
            right_embedding_list.append(right_embeddings)


        right_sequencial_embeddings = torch.stack(right_embedding_list) # Shape: (T, N, F)


        # Filtering imbalanced data only during training
        if self.filter_it and self.training:
            raise NotImplementedError("Filtering may be broken, please check it")
            # get filters
            left_filter, right_filter = get_filters_for_imbalance(dataBatchList)

            # filter it
            right_sequencial_embeddings[~right_filter] = 0

            # align non-zero embeddings to the right
            # TODO: Insert assert statement to make sure it is working as expectedly            
            right_sequencial_embeddings = align_embeddings_to_right(right_sequencial_embeddings, right_filter)

        if self.causal:
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(right_sequencial_embeddings.shape[0])
        else:
            causal_mask = None

        both_sequencial_embeddings = right_sequencial_embeddings
        # shape = (T, N, F)
        

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
            
        action_prediction = self.mlp(both_out)

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
        
        right_actions = self.head(right_embeddings)

        right_actions = right_actions.reshape(bs, self.temporal_length, self.num_actions).permute(1, 0, 2)

        return right_actions


class ActionRecognitionEncoderDecoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        assert args.merge_pred == "late"
        self.encoder = ActionRecognition(args)

        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(args.out_channels, args.num_heads_tr, args.dim_feedforward, args.dropout_tr), args.num_of_temp_layers)

        self.left_mlp = nn.Linear(args.out_channels, args.num_action_classes)
        self.right_mlp = nn.Linear(args.out_channels, args.num_action_classes)


    def forward(self, dataBatchList: List[Batch]):
        
        encoded = dataBatchList[:15]

        decoded = dataBatchList[15:]

        with torch.no_grad():
            encoded = self.encoder(encoded) # Bu class kullanılırken ActionRecognition class ı modifiye edilmeli. Öyle ki burada embedding return edilsin

        # No mask for decoder

        decoded = self.decoder(decoded, encoded)

        both_out_left, both_out_right = torch.tensor_split(decoded, 2, dim=1)

        both_out_ml = torch.cat([both_out_left, both_out_right], dim=-1)
        
        action_prediction = torch.cat((self.left_mlp(both_out_ml), self.right_mlp(both_out_ml)), dim=1)

        return action_prediction      




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