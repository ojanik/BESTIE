"""from torch.utils.data import Dataset
import torch


class SimpleDataset(Dataset):
    def __init__(self,input_data,flux_vars,sample_weights,norm_weights,additional_kwargs=[],kwargs_values=[]):
        
        assert len(input_data) == len(sample_weights)
        assert len(input_data) == len(norm_weights)
        #assert len(input_data) == len(fluxless_weights)

        for key in flux_vars.keys():
            assert len(input_data) == len(flux_vars[key])

        self.input_data = input_data
        self.flux_vars = flux_vars
        self.sample_weights = sample_weights
        self.norm_weights = norm_weights

        self.additional_kwargs = additional_kwargs
        self.kwargs_values = kwargs_values
        

    def __getitem__(self, idx):
        #data = {"reco_energy": self.data["energy_truncated"][idx],
        #       "reco_zenith": self.data["zenith_MPEFit"][idx]}
        input_data = torch.tensor([self.input_data[:,i][idx] for i in range(len(self.input_data[0]))])
        
        aux = {}
        for key in self.flux_vars.keys():
            aux[key] = self.flux_vars[key][idx]

        sample_weights = self.sample_weights[idx]
        norm_weights = self.norm_weights[idx]

        kwargs = {}

        for key in self.additional_kwargs:
            kwargs[key] = self.kwargs_values[key][idx]

        return input_data, aux, sample_weights, norm_weights, kwargs
    
    def __len__(self):
        return len(self.input_data)
"""