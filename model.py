from torch import nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten=nn.Flatten()
        self.network=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x=self.flatten(x)
        logits=self.network(x)
        return logits
    
def test_accuracy(models,dataloader,device='cpu'):
    n_corrects=0

    models=models.to(device)
    models.eval()
    for image_batch,label_batch in dataloader:
        image_batch=image_batch.to(device)
        label_batch=label_batch.to(device)

        with torch.no_grad():
            logits_batch=models(image_batch)

            predict_batch=logits_batch.argmax(dim=1)
            n_corrects+=(label_batch==predict_batch).sum().item()

    accuracy=n_corrects/len(dataloader.dataset)
    return accuracy

def train(models,dataloader,loss_fn,optimiser,device='cpu'):
    models.to(device)
    models.train()
    for image_batch,label_batch in dataloader:
        logits_batch=models(image_batch)
        image_batch=image_batch.to(device)
        label_batch=label_batch.to(device)

        loss=loss_fn(logits_batch,label_batch)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    return loss.item()

def test(models,dataloader,loss_fn,device='cpu'):
    models.to(device)
    loss_total=0.0
    models.eval()
    for image_batch,label_batch in dataloader:
        image_batch=image_batch.to(device)
        label_batch=label_batch.to(device)
    
        with torch.no_grad():
            logits_batch=models(image_batch)

        loss=loss_fn(logits_batch,label_batch)
        loss_total +=loss.item()

    return loss_total/len(dataloader)