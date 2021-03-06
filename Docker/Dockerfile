FROM nvcr.io/nvidia/pytorch:21.06-py3

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      curl \
      sox \
      libsox-dev \
      libsox-fmt-all \
      ca-certificates \
      sudo \
      git \
      bzip2 \
      git \
      wget \
      vim \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

RUN git clone https://github.com/take-koshizuka/continual-VC.git

WORKDIR /app/continual-VC

RUN pip install -r requirements.txt

CMD ["/bin/bash"]
