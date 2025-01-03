import numpy as np
import torch
import numpy
from collections import OrderedDict
from torch.utils import data
import esm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_embeddings_dim=320
esm_model, alphabet = esm.pretrained.load_model_and_alphabet("./ESM_models/esm2_t6_8M_UR50D.pt")
batch_converter = alphabet.get_batch_converter()
esm_model.eval()
pre_defined_window = 17
pre_half_size = pre_defined_window // 2
def get_data_mess(file_name):

    allproID_feas_labels = OrderedDict()
    with open('./lists/{0}.list'.format((file_name),'r')) as fr_dataname:  

        all_proID=[]
        for proID in fr_dataname:
            proID = proID.strip()

            all_pc_feas=[]
            all_pssm_feas=[]
            temp_join_pc_feas=[]
            temp_join_pssm_feas=[]
            if proID:
                allproID_feas_labels[proID] = {}
                all_proID.append(proID)

                with open('seqs/{0}.txt'.format(proID), 'r') as fr_proseqs:
                    lines = fr_proseqs.readlines()
                first_line = lines[0].strip()
                eachlineinfo = first_line.split(',')
                seq=eachlineinfo[0]
                CSlabel=int(eachlineinfo[1])
                ISlabel=int(eachlineinfo[2])
                NFSlabel=int(eachlineinfo[3])
                thisTemp = [(proID, seq)]

                batch_labels, batch_strs, batch_tokens = batch_converter(thisTemp)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

                with torch.no_grad():
                    results = esm_model(batch_tokens, repr_layers=[6])
                token_representations = results["representations"][6]
                temp_fea_embeddings = []
                temp_fea_embeddings = torch.FloatTensor(temp_fea_embeddings).view(-1, 320)
                for idx, tokens_len in enumerate(batch_lens):
                    esm_representation = token_representations[idx, 1: tokens_len - 1]
                    temp_fea_embeddings = torch.cat([temp_fea_embeddings, esm_representation], 0)
                fea_embeddings = temp_fea_embeddings[13 - pre_half_size : 13 + pre_half_size + 1, :]

                with open('feas/PCfeas/{0}.txt'.format(proID), 'r') as fr_feas:
                    for everyline in fr_feas:
                        everyline = everyline.strip()
                        if everyline:
                            everyline = [int(i) for i in everyline.split(',')]
                            temp_join_pc_feas.append(everyline)
                temp_join_pc_feas = np.array(temp_join_pc_feas)
                part_pc_feas = temp_join_pc_feas[13 - pre_half_size : 13 + pre_half_size + 1, :]
                all_pc_feas = part_pc_feas.flatten()

                with open('feas/PSSMfeas/{0}.pssm'.format(proID), 'r') as fr_feas:
                    for everyline in fr_feas:
                        everyline = everyline.strip()
                        if everyline:
                            everyline = [int(i) for i in everyline.split(',')]
                            temp_join_pssm_feas.append(everyline)
                temp_join_pssm_feas = np.array(temp_join_pssm_feas)
                part_pssm_feas = temp_join_pssm_feas[13 - pre_half_size: 13 + pre_half_size + 1, :]
                all_pssm_feas = part_pssm_feas.flatten()

                allproID_feas_labels[proID]['pc_feas'] = all_pc_feas
                allproID_feas_labels[proID]['pssm_feas'] = all_pssm_feas
                allproID_feas_labels[proID]['esm_feas'] = fea_embeddings.tolist()
                allproID_feas_labels[proID]['CS_labels'] = CSlabel
                allproID_feas_labels[proID]['IS_labels'] = ISlabel
                allproID_feas_labels[proID]['NFS_labels'] = NFSlabel

    return allproID_feas_labels, all_proID

class dataSet(data.Dataset):
    def __init__(self,file_name):
        self.allproID_features_labels, self.allproteinID = get_data_mess(file_name)

    def __getitem__(self, samp_idx):

        this_proID=self.allproteinID[samp_idx]
        this_esm_feas = numpy.array(self.allproID_features_labels[this_proID]['esm_feas'])
        this_pc_feas = numpy.array(self.allproID_features_labels[this_proID]['pc_feas'])
        this_pssm_feas = numpy.array(self.allproID_features_labels[this_proID]['pssm_feas'])
        this_labels_CS = self.allproID_features_labels[this_proID]['CS_labels']
        this_labels_IS = self.allproID_features_labels[this_proID]['IS_labels']
        this_labels_NFS = self.allproID_features_labels[this_proID]['NFS_labels']

        return this_esm_feas, this_pc_feas,this_pssm_feas
    def __len__(self):
        return len(self.allproteinID)

