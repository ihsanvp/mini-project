import torch
import constants

from dataloaders.redwood import RedwoodDataLoader
from datasets.redwood import RedwoodDataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = RedwoodDataset(
        constants.REDWOOD_METADATA_FILE,
        root="data",
        device=device,
    )
    dataloader = RedwoodDataLoader(
        dataset,
        data_root="data",
        batch_size=1,
        shuffle=False,
        max_data_size=constants.GB * 50,
        data_clear_padding=constants. GB * 5
    )

    it = iter(dataloader)

    batch = next(it)
    batch = next(it)
    batch = next(it)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
