from tqdm import tqdm
import constants
import json
import requests  # type: ignore
import cv2
import torch
import numpy as np
import open3d as o3d

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
        self.root = root
        self.device = device
        self.video_frames = video_frames
        self.ids = self._load_ids()

    def __getitem__(self, index):
        id, category = self.ids[index]

        self._download(id)

        f = self._get_files(id)
        video = self._load_video(f["video"])
        vert, face = self._load_mesh(f["mesh"])

        return video, vert, face, category

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
                    print(f"\tSkipping {metadata[t].name}")
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
                    with tqdm.wrapattr(open(filepath, "wb"), "write", miniters=1, desc=f"Downloading {filename}", total=int(r.headers.get('content-length', 0)), bar_format="        {l_bar}{bar:30}| {n_fmt}/{total_fmt} ") as fout:
                        for chunk in r.iter_content(CHUNK_SIZE):
                            fout.write(chunk)
                    # total = int(r.headers.get("Content-Length", 1))
                    # downloaded = 0
                    # percent = 0

                    # with open(filepath, "wb") as f:
                    #     for chunk in r.iter_content(CHUNK_SIZE):
                    #         if chunk:
                    #             f.write(chunk)

                    #             downloaded += len(chunk)
                    #             percent = round(downloaded / total * 100, 2)
                    #             filled = int((percent / 100) * BAR_SIZE)
                    #             remaining = BAR_SIZE - filled
                    #             bar = "[" + (filled * PROGRESS_FILLED) + \
                    #                 (remaining * PROGRESS_EMPTY) + "]"

                    #             sys.stdout.write(
                    #                 f"\r\tDownloading {filename} {bar} {percent}% ")
                    #             sys.stdout.flush()
                    # print(f"\tComplete {url}\t{humanize.naturalsize(total)}")
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

    def _load_mesh(self, path):
        mesh = o3d.io.read_triangle_mesh(str(path))

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        return torch.tensor(vertices, device=self.device), torch.tensor(faces, device=self.device)
