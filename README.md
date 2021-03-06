# Fine-tuning pre-trained voice conversion model for adding new target speakers with limited data

This repo contains code for our INTERSPEECH 2021 paper: [Fine-Tuning Pre-Trained Voice Conversion Model for Adding New Target Speakers with Limited Data](https://www.isca-speech.org/archive/interspeech_2021/koshizuka21_interspeech.html)

## Setup

1. Install Docker.
   * Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.
   * Setup running [Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

2. Clone this repository and `cd` into it.
    ```bash
    git clone https://github.com/take-koshizuka/continual-VC.git
    ```
3. Build the Docker image using Docker/Dockerfile
    ```bash
    cd Docker && docker build . -t env_vc
    ```
4. Run the container with `-it` flag. **All subsequent steps should be executed on the container.**
    ``` bash
    docker run --name env --runtime=nvidia -it env_vc
    ```

5. Download [CMU ARCTIC Database](http://www.festvox.org/cmu_arctic/) and [pre-trained vq-wav2vec parameters](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) using the script `continual-VC/download.py`.
    ```bash
    python3 download.py [-root] /path/to/root
    ```
    --root: Path to the directory where the dataset and the checkpoint are found or downloaded. (default: ./)

6. (Optional)  Download [trained decoder parameters](https://github.com/take-koshizuka/continual-VC/releases/tag/v1.0.0). 
    ```bash
    wget https://github.com/take-koshizuka/continual-VC/releases/download/v1.0.0/checkpoints.zip && unzip -u checkpoints.zip && rm checkpoints.zip
    ```

## 2. Training
### 2.1 Pre-training with VC task
```bash
python3 train.py -c config/train_baseline493.json -d checkpoints/baseline493
```

The path of the pre-trained model is set in configuration file.

### 2.2 Standard fine-tuning
```bash
python3 train.py -c config/train_fine.json -d checkpoints/fine-tuning
```

### 2.3 Rehearsal
```bash
python3 train.py -c config/train_reh.json -d checkpoints/reh
```

### 2.4 Pseudo-rehearsal
1. Generating pseudo speech dataset
   ```bash
   python3 generate_pseudo_data.py -c config/generate_pseudo_speech.json -p checkpoints/baseline493/best-model.pt -d pseudo_speech
   ```
   You can skip this step by downloading [the pseudo speech data](https://github.com/take-koshizuka/continual-VC/releases/tag/v1.0.0). 
    ```bash
    wget https://github.com/take-koshizuka/continual-VC/releases/download/v1.0.0/pseudo_speech.zip && unzip -u pseudo_speech.zip && rm pseudo_speech.zip
    ```

2. Training 
    ```bash
    python3 train_preh.py -c config/train_preh.json -d checkpoints/preh
    ```

## 3. Testing (Conversion)
WER and CER are computed with  [Transformer-based ASR model](https://zenodo.org/record/3966501#.YP-znlMzZwo) in our paper. However, in this implementation, we use [wav2vec2.0 model](https://pytorch.org/audio/stable/models.html#wav2vec2-large-lv60k) for simplicity of environment setup.

### 3.1 Evaluation for the pre-trained target speakers (aew, lnh, awb, ljm)
```bash
python3 convert.py -c config/convert_pre.json -p /path/to/checkpoints -d /path/to/output_dir
```

### 3.2 Evaluation for the fine-tuned target speakers (rms, slt)
```bash
python3 convert.py -c config/convert_fine.json -p /path/to/checkpoints -d /path/to/output_dir
```

## Citation
```bib
@inproceedings{koshizuka21_interspeech,
  author={Takeshi Koshizuka and Hidefumi Ohmura and Kouichi Katsurada},
  title={{Fine-Tuning Pre-Trained Voice Conversion Model for Adding New Target Speakers with Limited Data}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={1339--1343},
  doi={10.21437/Interspeech.2021-244}
}
```
