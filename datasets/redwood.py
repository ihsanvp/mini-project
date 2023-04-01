import constants
import json
import sys
import requests  # type: ignore
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset  # type: ignore

BASE_URL = "https://redwood-3dscan.b-cdn.net"

CHUNK_SIZE = 32768
BAR_SIZE = 20
PROGRESS_FILLED = "▓"
PROGRESS_EMPTY = " "

EXTENSIONS = {
    "mesh": ".ply",
    "video": ".mp4",
}


class RedwoodDataset(Dataset):
    def __init__(
        self,
        metadata,
        root="data",
        device=torch.device("cpu"),
        video_frames=1000
    ):
        self.metadata_file = metadata
        self.ids = self._load_ids()
        self.root = root
        self.device = device
        self.video_frames = video_frames

    def __getitem__(self, index):
        id, category = self.ids[index]

        self._download(id)

        f = self._get_files(id)
        v = self._load_video(f["video"])

        return v, category

    def __len__(self):
        return len(self.ids)

    def _load_ids(self):
        with open(self.metadata_file, "r") as f:
            m = json.load(f)

            meshes = m["meshes"]
            rgbds = m["rgbds"]
            categories = m["categories"]

            def find_category(id):
                for c in categories:
                    if id in categories[c]:
                        return c
                return None

            return [(x, find_category(x))
                    for x in meshes if x in rgbds and find_category(x) != None]

    def _get_files(self, id):
        return {
            "mesh": constants.BASE_DIR / self.root / "mesh" / f"{id}.ply",
            "video": constants.BASE_DIR / self.root / "video" / f"{id}.mp4",
        }

    def _build_url_object(self, id, t):
        if t in EXTENSIONS:
            ext = EXTENSIONS[t]
            return (f"{BASE_URL}/{t}/{id}{ext}", f"{id}{ext}", t)
        else:
            raise ValueError()

    def _download(self, id, skip_if_exists=True):
        print(f"Loading {id}...")

        location = constants.BASE_DIR / self.root
        metadata = self._get_files(id)
        files = [
            self._build_url_object(id, "mesh"),
            self._build_url_object(id, "video"),
        ]

        if location.is_dir() != True:
            location.mkdir(parents=True)

        if skip_if_exists:
            for t in metadata:
                if metadata[t].exists():
                    print(f"\tDownload skipped {metadata[t].name}")
                    files.remove(self._build_url_object(id, t))

            if len(files) == 0:
                return True

        if location.is_dir() != True:
            location.mkdir(parents=True)

        for each in files:
            try:
                url, filename, folder = each
                filepath = location / folder / filename

                if filepath.parent.is_dir() != True:
                    filepath.parent.mkdir(parents=True)

                r = requests.get(url, stream=True)
                if r.ok:
                    total = int(r.headers.get("Content-Length", 1))
                    downloaded = 0
                    percent = 0

                    with open(filepath, "wb") as f:
                        for chunk in r.iter_content(CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)

                                downloaded += len(chunk)
                                percent = round(downloaded / total * 100, 2)
                                filled = int((percent / 100) * BAR_SIZE)
                                remaining = BAR_SIZE - filled
                                bar = "[" + (filled * PROGRESS_FILLED) + \
                                    (remaining * PROGRESS_EMPTY) + "]"

                                sys.stdout.write(
                                    f"\r\tDownloading {filename} {bar} {percent}% ")
                                sys.stdout.flush()
                    print(f"\tComplete {url}")
                else:
                    print("Download failed")
            except:
                url, filename, folder = each
                filepath = location / folder / filename

                if filepath.exists():
                    filepath.unlink()

                print("Download failed due to Exception")

    def _load_video(self, path):
        cap = cv2.VideoCapture(str(path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        f = 0
        ret = True
        buffer = []

        frames = total_frames if total_frames < self.video_frames else self.video_frames

        while f < frames and ret:
            ret, frame = cap.read()
            buffer.append(frame)
            f += 1

        if total_frames < self.video_frames:
            for _ in range(self.video_frames - total_frames):
                buffer.append(np.zeros((height, width, 3)))

        cap.release()
        return torch.tensor(np.array(buffer, dtype=np.uint8), device=self.device)
