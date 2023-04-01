import torch

import constants

from datasets.redwood import RedwoodDataset
from torch.utils.data import DataLoader


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = RedwoodDataset(constants.REDWOOD_METADATA_FILE, device=device)
    dataloader = DataLoader(dataset, batch_size=5,
                            shuffle=False)

    batch = next(iter(dataloader))

    print(batch[0])
    print(batch[1])


if __name__ == "__main__":
    main()
