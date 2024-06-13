from torch.utils.data import Dataset
import torch


class SimpleDataset(Dataset):
    def __init__(self,input_data):
        self.input_data = input_data

        
        
    def __getitem__(self, idx):
        #data = {"reco_energy": self.data["energy_truncated"][idx],
        #       "reco_zenith": self.data["zenith_MPEFit"][idx]}
        data = torch.tensor([self.input_data[:,0][idx],self.input_data[:,1][idx]])
        
        aux = {"true_energy": self.input_data[:,2][idx],
                'mceq_pr_H4a_SIBYLL23c': self.input_data[:,6][idx],
                'mceq_pr_GST4_SIBYLL23c': self.input_data[:,7][idx],
                'powerlaw': self.input_data[:,3][idx],
                'mceq_conv_H4a_SIBYLL23c': self.input_data[:,4][idx],
                'mceq_conv_GST4_SIBYLL23c':self.input_data[:,5][idx]}
        
        return data, aux
    
    def __len__(self):
        return len(self.input_data)