# Under Construction...

# Fine-tuning pre-trained voice conversion model for adding new target speakers with limited data

This is the official implementation of the paper "Fine-tuning pre-trained voice conversion model for adding new target speakers with limited data".


# Setup

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


## 2. Training

### 2.1 Pre-training with VC task
```bash
python3 train.py -c config/train_pre.json -d checkpoints/pre
```

### 2.2 Standard fine-tuning
```bash
python3 train.py -c config/train_fine.json -d checkpoints/fine-tuning
```
The path of the pre-trained model is set in configuration file.

### 2.3 Rehearsal
```bash
python3 train.py -c config/train_reh.json -d checkpoints/reh
```

### 2.4 Pseudo-rehearsal
1. generate pseudo speech dataset
   ```bash
   python3 generate_pseudo_data.py
   ```
2. 

## 3. Testing (Conversion)

