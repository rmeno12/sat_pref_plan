import datetime

import torch
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import Mean

from sat_pref_plan.data.modified import ModifiedPyramidPatchPatchDataset
from sat_pref_plan.models.convnet import ConvClassifier
from sat_pref_plan.run import utils


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    im1 = "./eer_lawn_unmasked.jpg"
    im2 = "./eer_lawn_moremasked.jpg"

    patch_size = 64
    num_levels = 3
    dataset = ModifiedPyramidPatchPatchDataset(
        im1, im2, patch_size, num_levels=num_levels, stride=16
    )
    train, val = random_split(
        dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )
    batch_size = 32
    tloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
    vloader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Training size: {len(train)}, Validation size: {len(val)}")

    learning_rate = 0.01
    epochs = 500
    model = ConvClassifier(1, in_channels=3 * num_levels).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss()

    # data = next(iter(tloader))
    # # while not data[1].any():
    # #     data = next(iter(tloader))
    # print(data[0].shape, data[1].shape)
    # print(data[0].dtype, data[1].dtype)
    # print(data[1])
    # p = model(data[0].to(device))
    # print(p.shape)
    # print(p.dtype)

    tlosses = []
    vlosses = []
    # tf1s = []
    # vf1s = []
    for t in range(epochs):
        model.train(True)
        # tf1 = BinaryF1Score(device=device)
        tfl = Mean(device=device)
        for batch, (X, y) in enumerate(tloader):
            X, y = X.to(device), y.to(device)

            # predict
            pred = model(X)
            loss = loss_fn(pred, y[:, None].float())
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
            loss = loss_fn(pred, y[:, None].float())
            # vf1.update(pred[:, 0], y[:, 0])
            vl.update(loss)
        vlosses.append(vl.compute().item())
        # vf1s.append(vf1.compute().item())

        if t % 10 == 9:
            print(f"Epoch {t+1}")
            print(f"Val Loss: {vlosses[-1]}")
            print(f"Train Loss: {tlosses[-1]}")

    saveroot = "results/conv_modpyr/"
    stamp = str(datetime.datetime.now()).replace(" ", "_")

    utils.plot_loss(tlosses, vlosses, saveroot + f"{stamp}_loss.png")

    torch.save(model.state_dict(), saveroot + f"{stamp}.pt")
    return


if __name__ == "__main__":
    main()
