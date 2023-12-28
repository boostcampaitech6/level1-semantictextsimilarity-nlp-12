import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return {
                'input_ids' : torch.tensor(self.inputs[idx][0]), 
                'attention_mask' : torch.tensor(self.inputs[idx][1])
            }
        else:
            return {
                'input_ids' : torch.tensor(self.inputs[idx][0]), 
                'attention_mask' :torch.tensor(self.inputs[idx][1]), 
                'target' : torch.tensor(self.targets[idx])
            }

    def __len__(self):
        return len(self.inputs)