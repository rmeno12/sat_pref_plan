import models
import datetime
import datasets
from torch.utils.data import DataLoader, random_split
import torch
from torcheval.metrics import BinaryF1Score, Mean
from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

im1 = "./eer_lawn_unmasked.jpg"
im2 = "./eer_lawn_moremasked.jpg"

patch_size = 32
dataset = datasets.MatchingPatchDataset(im1, im2, patch_size)
train, val = random_split(
    dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)
tloader = DataLoader(train, batch_size=32, shuffle=True)
vloader = DataLoader(val, batch_size=32, shuffle=False)
print(f"Training size: {len(train)}, Validation size: {len(val)}")

learning_rate = 0.01
epochs = 500
model = models.UNet(3, 3).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


# data = next(iter(tloader))
# print(data[0].shape, data[1].shape)
# p = model(data[0].to(device))
# print(p.shape)

tlosses = []
vlosses = []
tf1s = []
vf1s = []
for t in range(epochs):
    model.train(True)
    tf1 = BinaryF1Score(device=device)
    tfl = Mean(device=device)
    for batch, (X, y) in enumerate(tloader):
        X, y = X.to(device), y.to(device)

        # predict
        pred = model(X)
        loss = loss_fn(pred, y)
        # tf1.update(pred[:, 0], y[:, 0])
        tfl.update(loss)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tlosses.append(tfl.compute().item())
    # tf1s.append(tf1.compute().item())

    model.train(False)
    # vf1 = BinaryF1Score(device=device)
    vl = Mean(device=device)
    for batch, (X, y) in enumerate(vloader):
        X, y = X.to(device), y.to(device)

        # predict
        pred = model(X)
        loss = loss_fn(pred, y)
        # vf1.update(pred[:, 0], y[:, 0])
        vl.update(loss)
    vlosses.append(vl.compute().item())
    # vf1s.append(vf1.compute().item())

    if t % 10 == 9:
        print(f"Epoch {t+1}")
        print(f"Val Loss: {vlosses[-1]}")
        # print(f"Val F1 Score: {vf1s[-1]}")


plt.plt(tlosses, label="Training Loss")
plt.plt(vlosses, label="Validation Loss")
plt.legend()
plt.savefig("results/unet_matchpatch/loss.png")

stamp = str(datetime.datetime.now()).replace(" ", "_")
torch.save(model.state_dict(), f"results/unet_matchpatch/{stamp}.pt")
