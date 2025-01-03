import os
from torch.utils.data import DataLoader
from model import *
from loadData import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fixed_window = 17
dim_esm_embeddings=320
input_esm_dim = dim_esm_embeddings * fixed_window
dim_pc_feas=10
dim_pssm_feas=20

def for_TEST_experi(model, loader):
    model.eval()
    all_CS_preds = []
    for batch_idx, (ESM2feas, pcfeas, pssmfeas) in enumerate(loader):

        with torch.no_grad():

            temp_feas_ESM2 = torch.autograd.Variable(ESM2feas.to(torch.float32).to(device))
            feas_ESM2 = temp_feas_ESM2.view(-1, input_esm_dim)

            temp_pcfeas_computed = torch.autograd.Variable(pcfeas.to(torch.float32).to(device))
            feas_pc = temp_pcfeas_computed.view(-1, fixed_window, dim_pc_feas)

            temp_pssmfeas_computed = torch.autograd.Variable(pssmfeas.to(torch.float32).to(device))
            feas_pssm = temp_pssmfeas_computed.view(-1, fixed_window, dim_pssm_feas)

        CS_preds = model(feas_ESM2, feas_pc, feas_pssm)
        all_CS_preds.append(CS_preds.data.cpu().numpy())

    all_CS_preds = np.concatenate(all_CS_preds, axis=0).flatten()
    return all_CS_preds

print('===========================')
test_dataset = dataSet('test')
print('test_dataset is done')

myModel = MainModel().to(device)

print('===========================')
print('Model Construction is done')

test_loader = DataLoader(test_dataset, batch_size=256, pin_memory=True, num_workers=10, drop_last=False)
print('===========================')
print('test_loader is done')
print('===========================')

all_CS_preds = []
all_CS_labels = []
all_IS_labels = []

path_dir = "./CSModel/"
myModel.load_state_dict(torch.load(os.path.join(path_dir, 'model_weights_44.dat')))

all_CS_preds = for_TEST_experi (model=myModel, loader=test_loader)
print("test prediction is done!")
path_dir = "./pred_results/"
if not os.path.exists(path_dir):
    os.makedirs(path_dir)

with open('pred_results/test.txt', 'w', encoding="utf-8") as fw_cs:
    for idx, values in enumerate(all_CS_preds):
        fw_cs.write(str(round(values, 5))+"\n")
fw_cs.close()