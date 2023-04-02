import random
import humanize

from torch.utils.data import DataLoader
from constants import BASE_DIR, GB, MB


class RedwoodDataLoader():
    def __init__(
        self,
        dataset,
        data_root="data",
        batch_size=1,
        shuffle=False,
        max_data_size=GB,
        data_clear_padding=GB / 2,
        workers=0
    ):
        self.data_root = BASE_DIR / str(data_root)
        self.max_data_size = max_data_size
        self.data_clear_padding = data_clear_padding
        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)

    def __iter__(self):
        for data in self.loader:
            yield data

            exceeded = self._root_data_diff()
            if exceeded > 0:
                self._clear_root(exceeded)

    def __len__(self):
        return len(self.loader)

    def _root_data_diff(self) -> int:
        return sum(f.stat().st_size for f in self.data_root.glob('**/*') if f.is_file()) - self.max_data_size

    def _clear_root(self, size_exceeded):
        files = [(f, f.stat().st_size)
                 for f in self.data_root.glob("**/*") if f.is_file()]
        clear_target = size_exceeded + self.data_clear_padding

        cleared = 0
        count = 0

        while cleared < clear_target:
            f = random.choice(files)
            f[0].unlink()
            files.remove(f)
            cleared += f[1]
            count += 1

        print(f"Cleared {count} files. ({humanize.naturalsize(cleared)})")
