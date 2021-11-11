FROM ubuntu:18.04 as intermediate

LABEL maintainer="Yuncheng Wu <lemonwyc@gmail.com>"

RUN apt-get update && apt-get upgrade -y && \
        apt-get install -y --no-install-recommends \
        git \
        sqlite3 \
        curl \
        sudo \
        unzip \
        vim \
        wget \
        zip \
        ssh

#Accept input argument
ARG SSH_PRIVATE_KEY

#Pass the content of the private key into the container
RUN mkdir /root/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa

#Github requires a private key with strict permission settings
RUN chmod 600 /root/.ssh/id_rsa

#Add Github to known hosts
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# Clone pivot and init submodules
WORKDIR /opt
RUN git config --global http.sslVerify false
RUN git clone git@github.com:nusdbsystem/pivot.git && \
    cd pivot/third_party/ && \
    git submodule update --init --recursive

FROM ubuntu:18.04

COPY --from=intermediate /opt/pivot /opt/pivot

RUN apt-get update && apt-get upgrade -y && \
        apt-get install -y --no-install-recommends \
        libgmp-dev \
        libboost-dev \
        libboost-all-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libcurl4-openssl-dev \
        build-essential \
        ca-certificates \
        gdb \
        git \
        libjsoncpp-dev \
        liblog4cpp5-dev \
        libprotobuf-c-dev \
        libprotobuf-dev \
        libssl-dev \
        libtool \
        m4 \
        protobuf-compiler \
        python \
        python3 \
        python3-pip \
        libgoogle-glog-dev \
        libgtest-dev \
        texinfo \
        yasm \
        automake \
        libsodium-dev \
        libcrypto++-dev \
        libcrypto++-doc \
        libcrypto++-utils \
        libgmp3-dev \
        autoconf \
        pkg-config \
        wget \
        && \
        pip3 install requests && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# Install
RUN dpkg-reconfigure dash

# Upgrade cmake version to 3.19.7
RUN mkdir ~/temp && \
    cd ~/temp && \
    wget https://cmake.org/files/v3.19/cmake-3.19.7.tar.gz && \
    tar -xzvf cmake-3.19.7.tar.gz && \
    cd cmake-3.19.7/ && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install && \
    hash -r && \
    cmake --version

# Install protobuf v3.14.0
RUN cd ~/temp && \
    pwd && \
    wget https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protobuf-all-3.14.0.tar.gz && \
    tar -xzvf protobuf-all-3.14.0.tar.gz && \
    cd protobuf-3.14.0 && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    protoc --version && \
    cd cmake && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

# Install third_party threshold partially homomorphic encryption library
WORKDIR /opt/pivot
RUN cd third_party/libhcs && \
    cmake . && \
    make && \
    make install

# Install third_party Pivot-SPDZ library
WORKDIR /opt/pivot
RUN cd third_party/Pivot-SPDZ && \
    make mpir && \
    bash fast-make.sh && \
    ./compile.py Programs/Source/vfl_decision_tree.mpc && \
    ln -s /opt/pivot/third_party/Pivot-SPDZ/local/lib/libmpir* /usr/local/lib/

# Install third_party libscapi library
WORKDIR /opt/pivot
RUN cd third_party/libscapi && \
    make

# Generate protobuf messages
RUN cd /opt/pivot/src/include/ && \
    SRC_DIR=meta/ && \
    DST_DIR=protobuf && \
    protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/*.proto

# Build pivot
WORKDIR /opt/pivot
RUN mkdir log && \
    export PATH="$PATH:$HOME/.local/bin" && \
    export PATH="$PATH:/usr/local/bin" && \
    mkdir build && \
    cmake -Bbuild -H. && \
    cd build/ && \
    make

# Define working directory.
WORKDIR /opt