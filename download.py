import os
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)
import csv
import argparse
import requests
import tqdm


FILE_TEXT = "txt.done.data"
FOLDER_TEXT = "etc"
EXT_AUDIO = ".wav"
FOLDER_AUDIO = "wav"
FOLDER_IN_ARCHIVE = "ARCTIC"
FOLDER_MODEL_CHECKPOINT = "checkpoints"
_CHECKSUMS = {
    "http://festvox.org/cmu_arctic/packed/cmu_us_aew_arctic.tar.bz2":
    "4382b116efcc8339c37e01253cb56295",
    "http://festvox.org/cmu_arctic/packed/cmu_us_ahw_arctic.tar.bz2":
    "b072d6e961e3f36a2473042d097d6da9",
    "http://festvox.org/cmu_arctic/packed/cmu_us_aup_arctic.tar.bz2":
    "5301c7aee8919d2abd632e2667adfa7f",
    "http://festvox.org/cmu_arctic/packed/cmu_us_awb_arctic.tar.bz2":
    "280fdff1e9857119d9a2c57b50e12db7",
    "http://festvox.org/cmu_arctic/packed/cmu_us_axb_arctic.tar.bz2":
    "5e21cb26c6529c533df1d02ccde5a186",
    "http://festvox.org/cmu_arctic/packed/cmu_us_bdl_arctic.tar.bz2":
    "b2c3e558f656af2e0a65da0ac0c3377a",
    "http://festvox.org/cmu_arctic/packed/cmu_us_clb_arctic.tar.bz2":
    "3957c503748e3ce17a3b73c1b9861fb0",
    "http://festvox.org/cmu_arctic/packed/cmu_us_eey_arctic.tar.bz2":
    "59708e932d27664f9eda3e8e6859969b",
    "http://festvox.org/cmu_arctic/packed/cmu_us_fem_arctic.tar.bz2":
    "dba4f992ff023347c07c304bf72f4c73",
    "http://festvox.org/cmu_arctic/packed/cmu_us_gka_arctic.tar.bz2":
    "24a876ea7335c1b0ff21460e1241340f",
    "http://festvox.org/cmu_arctic/packed/cmu_us_jmk_arctic.tar.bz2":
    "afb69d95f02350537e8a28df5ab6004b",
    "http://festvox.org/cmu_arctic/packed/cmu_us_ksp_arctic.tar.bz2":
    "4ce5b3b91a0a54b6b685b1b05aa0b3be",
    "http://festvox.org/cmu_arctic/packed/cmu_us_ljm_arctic.tar.bz2":
    "6f45a3b2c86a4ed0465b353be291f77d",
    "http://festvox.org/cmu_arctic/packed/cmu_us_lnh_arctic.tar.bz2":
    "c6a15abad5c14d27f4ee856502f0232f",
    "http://festvox.org/cmu_arctic/packed/cmu_us_rms_arctic.tar.bz2":
    "71072c983df1e590d9e9519e2a621f6e",
    "http://festvox.org/cmu_arctic/packed/cmu_us_rxr_arctic.tar.bz2":
    "3771ff03a2f5b5c3b53aa0a68b9ad0d5",
    "http://festvox.org/cmu_arctic/packed/cmu_us_slp_arctic.tar.bz2":
    "9cbf984a832ea01b5058ba9a96862850",
    "http://festvox.org/cmu_arctic/packed/cmu_us_slt_arctic.tar.bz2":
    "959eecb2cbbc4ac304c6b92269380c81",
}

SPEAKERS = [
    "aew",
    "awb",
    "bdl",
    "clb",
    "ljm",
    "lnh",
    "rms",
    "slt"
]


def download_dataset(root, speaker):
    url = "cmu_us_" + speaker + "_arctic"
    ext_archive = ".tar.bz2"
    base_url = "http://www.festvox.org/cmu_arctic/packed/"
    url = os.path.join(base_url, url + ext_archive)
    root = os.fspath(root)

    basename = os.path.basename(url)
    root = os.path.join(root, FOLDER_IN_ARCHIVE)
    if not os.path.isdir(root):
        os.mkdir(root)
    archive = os.path.join(root, basename)

    basename = basename.split(".")[0]
    path = os.path.join(root, basename)
    if not os.path.isdir(path):
        if not os.path.isfile(archive):
            checksum = _CHECKSUMS.get(url, None)
            download_url(url, root, hash_value=checksum, hash_type="md5")
        extract_archive(archive)

    text = os.path.join(path, FOLDER_TEXT, FILE_TEXT)
    if not os.path.isfile(text):
        with open(text, "r") as text:
            walker = csv.reader(text, delimiter="\n")
            _walker = list(walker)

def download_model(root):
    root = os.path.join(root, FOLDER_MODEL_CHECKPOINT)
    if not os.path.isdir(root):
        os.mkdir(root)
    url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec.pt"
    filename = os.path.join(root, "vq-wav2vec.pt")

    if not os.path.isfile(filename):
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(filename, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()
        assert total_size_in_bytes == 0 or progress_bar.n == total_size_in_bytes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', help="Path to the directory where the dataset is found or downloaded.", default='./', type=str)
    args = parser.parse_args()

    download_model(args.root)
    for speaker in SPEAKERS:
        download_dataset(args.root, speaker)