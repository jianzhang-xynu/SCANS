import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import Conv2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fixed_window = 17 

esm_embeddings=320
input_esm_dim = esm_embeddings * fixed_window

dim1=20
dim2=10

TRF_input = 20
d_ff = 512
d_k = d_v = 128
n_heads = 8
n_layers = 4
TRF_output = 10

class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.IS_Conv_layer = IS_ConvLayer()
        self.IS_dim_Conv_out = 10
        self.IS_TRF_layer = Transformer(TRF_input, TRF_output)
        IS_comb_input = self.IS_dim_Conv_out + TRF_output
        self.ESM2 = nn.Sequential()
        self.ESM2.add_module('ESM2', FCL2layers(input_esm_dim, dim1, dim2))
        self.Conv_layer = ConvLayer()
        self.TRF_layer = Transformer(TRF_input, TRF_output)
        CS_comb_input = dim2 + TRF_output
        comb_input = IS_comb_input + CS_comb_input
        self.PredLayers = nn.Sequential()
        self.PredLayers.add_module('PredLayers', Predlayers(comb_input, 25, 12, 6))

    def forward(self, embedding_feas, pc_feas, PSSM_feas):
        IS_output_PC = self.IS_Conv_layer(pc_feas)
        IS_output_PSSM = self.IS_TRF_layer(PSSM_feas)
        IS_comb_feas = torch.cat((IS_output_PC, IS_output_PSSM), 1)
        output_ESM2 = self.ESM2(embedding_feas)
        output_PC = self.Conv_layer(pc_feas)
        output_PSSM = self.TRF_layer(PSSM_feas)
        CS_comb_feas = torch.cat((output_ESM2, output_PC), 1)
        CS_comb_feas = torch.cat((output_ESM2, output_PSSM), 1)
        comb_feas = torch.cat((IS_comb_feas, CS_comb_feas), 1)
        final_pred = self.PredLayers(comb_feas)
        return final_pred


class FCL2layers(nn.Module):
    def __init__(self, input_esm_dim, dim1, dim2):
        super(FCL2layers, self).__init__()
        self.linear1 = nn.Linear(input_esm_dim, dim1, bias=False)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(dim1, dim2)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        features = self.linear1(x)
        features = self.activation1(features)
        features = self.dropout1(features)
        features = self.linear2(features)
        features = self.activation2(features)
        features = self.dropout2(features)
        FCL2layers_output = features

        return FCL2layers_output

class Predlayers(nn.Module):
    def __init__(self, inputdim, pred_dim1, pred_dim2, pred_dim3):
        super(Predlayers, self).__init__()
        self.linear1 = nn.Linear(inputdim, pred_dim1, bias=False)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(pred_dim1, pred_dim2, bias=False)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(pred_dim2, pred_dim3, bias=False)
        self.activation3 = nn.ReLU()
        self.linear4 = nn.Linear(pred_dim3, 1)

    def forward(self, x):
        features = self.linear1(x)
        features = self.activation1(features)
        features = self.dropout1(features)
        features = self.linear2(features)
        features = self.activation2(features)
        features = self.linear3(features)
        features = self.activation3(features)
        features = self.linear4(features)
        Predlayers_output = torch.sigmoid(features)

        return Predlayers_output


class IS_ConvLayer(nn.Module):
    def __init__(self):
        super(IS_ConvLayer, self).__init__()

        self.conv1DLayers=nn.Sequential()
        self.conv1DLayers.add_module("conv1", nn.Conv1d(in_channels=10, out_channels=8, kernel_size=2, stride=1))
        self.conv1DLayers.add_module("norm1", nn.BatchNorm1d(8))
        self.conv1DLayers.add_module("ReLU1", nn.ReLU())
        self.conv1DLayers.add_module("Dropout1", nn.Dropout(0.5))
        self.conv1DLayers.add_module("Pooling1", nn.MaxPool1d(kernel_size=2, stride=1))
        self.conv1DLayers.add_module("conv2", nn.Conv1d(in_channels=8, out_channels=2, kernel_size=2, stride=1))
        self.conv1DLayers.add_module("norm2", nn.BatchNorm1d(2))
        self.conv1DLayers.add_module("ReLU2", nn.ReLU())
        self.conv1DLayers.add_module("Dropout2", nn.Dropout(0.5))
        self.conv1DLayers.add_module("Pooling2", nn.MaxPool1d(kernel_size=2, stride=1))
        self.fcnnLayers1=nn.Sequential()
        self.fcnnLayers1.add_module("linear1",nn.Linear(26, 5, bias=True))  
        self.fcnnLayers1.add_module("activation1", nn.ReLU())

        self.conv2DLayers=nn.Sequential()
        self.conv2DLayers.add_module("conv1",nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(2,2),stride=1))
        self.conv2DLayers.add_module("norm1", nn.PReLU())
        self.conv2DLayers.add_module("Dropout1", nn.Dropout(0.5))
        self.conv2DLayers.add_module("Pooling1", nn.MaxPool2d(kernel_size=(2,2), stride=1))
        self.conv2DLayers.add_module("conv2",nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(2,2),stride=1))
        self.conv2DLayers.add_module("ReLU2", nn.PReLU())
        self.conv2DLayers.add_module("Dropout2", nn.Dropout(0.5))
        self.conv2DLayers.add_module("Pooling2", nn.MaxPool2d(kernel_size=(2,2), stride=1))
        self.fcnnLayers2=nn.Sequential()
        self.fcnnLayers2.add_module("linear1",nn.Linear(312, 5, bias=True))  
        self.fcnnLayers2.add_module("activation1", nn.ReLU())

    def forward(self, PCfeas):
        feas_1D = torch.transpose(PCfeas, 1, 2)
        output_conv1D=self.conv1DLayers(feas_1D)
        output1 = self.fcnnLayers1(output_conv1D.view(-1,26))  
        feas_2D = PCfeas.unsqueeze(0)
        feas_2D = torch.transpose(feas_2D, 0, 1)
        output_conv2D=self.conv2DLayers(feas_2D)
        output2=self.fcnnLayers2(output_conv2D.view(-1,312))  
        out=torch.cat((output1,output2),dim=1)
        return out

class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()

        self.conv1DLayers=nn.Sequential()
        self.conv1DLayers.add_module("conv1", nn.Conv1d(in_channels=10, out_channels=8, kernel_size=2, stride=1))
        self.conv1DLayers.add_module("norm1", nn.BatchNorm1d(8))
        self.conv1DLayers.add_module("ReLU1", nn.ReLU())
        self.conv1DLayers.add_module("Dropout1", nn.Dropout(0.5))
        self.conv1DLayers.add_module("Pooling1", nn.MaxPool1d(kernel_size=2, stride=1))
        self.conv1DLayers.add_module("conv2", nn.Conv1d(in_channels=8, out_channels=2, kernel_size=2, stride=1))
        self.conv1DLayers.add_module("norm2", nn.BatchNorm1d(2))
        self.conv1DLayers.add_module("ReLU2", nn.ReLU())
        self.conv1DLayers.add_module("Dropout2", nn.Dropout(0.5))
        self.conv1DLayers.add_module("Pooling2", nn.MaxPool1d(kernel_size=2, stride=1))
        self.fcnnLayers1=nn.Sequential()
        self.fcnnLayers1.add_module("linear1",nn.Linear(26, 5, bias=True))  
        self.fcnnLayers1.add_module("activation1", nn.ReLU())

        self.conv2DLayers=nn.Sequential()
        self.conv2DLayers.add_module("conv1",nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(2,2),stride=1))
        self.conv2DLayers.add_module("norm1", nn.PReLU())
        self.conv2DLayers.add_module("Dropout1", nn.Dropout(0.5))
        self.conv2DLayers.add_module("Pooling1", nn.MaxPool2d(kernel_size=(2,2), stride=1))
        self.conv2DLayers.add_module("conv2",nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(2,2),stride=1))
        self.conv2DLayers.add_module("ReLU2", nn.PReLU())
        self.conv2DLayers.add_module("Dropout2", nn.Dropout(0.5))
        self.conv2DLayers.add_module("Pooling2", nn.MaxPool2d(kernel_size=(2,2), stride=1))
        self.fcnnLayers2=nn.Sequential()
        self.fcnnLayers2.add_module("linear1",nn.Linear(312, 5, bias=True))  
        self.fcnnLayers2.add_module("activation1", nn.ReLU())

    def forward(self, PCfeas):

        feas_1D = torch.transpose(PCfeas, 1, 2)
        output_conv1D=self.conv1DLayers(feas_1D)
        output1 = self.fcnnLayers1(output_conv1D.view(-1,26))
        feas_2D = PCfeas.unsqueeze(0)
        feas_2D = torch.transpose(feas_2D, 0, 1)
        output_conv2D=self.conv2DLayers(feas_2D)
        output2=self.fcnnLayers2(output_conv2D.view(-1,312)) 
        out=torch.cat((output1,output2),dim=1)
        return out

class Transformer(nn.Module):
    def __init__(self,d_model,TRF_output):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model)  
        self.projection = nn.Linear(d_model, TRF_output, bias=False)

    def forward(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_logits = self.projection(enc_outputs)
        mid_window_scores = dec_logits[:, fixed_window // 2, :]
        return mid_window_scores.squeeze(1)
class Encoder(nn.Module):
    def __init__(self,d_model):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)  
        self.layers = nn.ModuleList(EncoderLayer(d_model) for _ in range(n_layers))  

    def forward(self, enc_inputs):
        enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask()
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  
        pe = torch.zeros(max_len, d_model)  #
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  #
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self,d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1).to(device)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        output = self.layer_norm(output + residual)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.LayerNorm(d_model)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return (output + residual)

def get_attn_pad_mask():
    oriMask = [[False] * fixed_window for _ in range(fixed_window)]
    for row in range(len(oriMask[0])):
        for col in range(len(oriMask[1])):
            if row == fixed_window // 2 or col == fixed_window // 2:
                oriMask[row][col] = True

    pad_attn_mask = torch.Tensor(oriMask).bool().to(device)

    return pad_attn_mask.expand(1, fixed_window, fixed_window)


