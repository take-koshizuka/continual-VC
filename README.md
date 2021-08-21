# Under Construction...

# Fine-tuning pre-trained voice conversion model for adding new target speakers with limited data
---

This is the official implementation of the paper "Fine-tuning pre-trained voice conversion model for adding new target speakers with limited data".


# Setup

1. Install Docker.
   * Install NVIDIA Container Toolkit for GPU support.
   * Setup running Docker as a non-root user.

2. Clone this repository and `cd` into it.
    ```bash
    git clone https://github.com/take-koshizuka/continual-VC.git
    ```
3. Build the Docker image using Docker/Dockerfile
    ```bash
    cd Docker && docker build . -t env_vc
    ```
4. Run the container with `-it` flag. <u>All subsequent steps will be executed on the container. </u>
    ``` bash
    docker run --name env --runtime=nvidia -it env_vc
    ```

5. Download [CMU ARCTIC Database](http://www.festvox.org/cmu_arctic/) and [pre-trained vq-wav2vec parameters](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) using `continual-VC/download.py`.
    ```bash
    python3 download.py [-root] /path/to/root
    ```
    --root: Path to the directory where the dataset and the checkpoint are found or downloaded. (default: ./)


## 2. Training




## 3. Testing (Conversion)

