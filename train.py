import torch
from torch import nn 
from tqdm.auto import tqdm
from torch.optim import Adam, SGD
from model import Model
from torch.utils.data import DataLoader
from dataset import ImageDataset
from torchvision import transforms

def train(epochs, model, trainloader, optimizer, criterion, device):

    for epoch in tqdm(range(epochs)):
        print(f"===epoch: {epoch}")

        for rich_imgs, poor_imgs, labels in trainloader:
            rich_imgs = rich_imgs.to(device)
            poor_imgs = poor_imgs.to(device)
            labels = labels.to(device)

            # Get predicted labels
            outputs = model(rich_imgs, poor_imgs)

            loss = criterion(outputs.squeeze().float(), labels.float())
            loss.backward()

            optimizer.step()
    
    return model

"""
    Define model and optimizer and loss fucntion
"""
model = Model()
criterion = nn.BCELoss()
lr = 0.001
optimizer = Adam(model.parameters(), lr=lr)

"""
    Get Dataloader
"""
data_dir = 'rd-real-GAN-DM' #replace path here 

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageDataset(image_transform, data_dir, patch_size=32)
trainloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

"""
    Train
"""
if __name__ == "__main__":
    model = train(epochs, model, trainloader, optimizer, criterion, device)

