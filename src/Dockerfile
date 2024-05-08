FROM ultralytics/ultralytics:latest-jetson

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    vim \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gnupg \
    locales \
    lsb-release \
    software-properties-common \
    tmux \
 && rm -rf \var/lib/apt/lists/*


# Check existing locale settings (useful for debugging; outputs to build logs)
RUN locale

# Install locales package and generate en_US.UTF-8
RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Set the environment variables for locale
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Verify locale settings
RUN locale


# Install necessary packages for managing repositories
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe

# Install curl and add ROS 2 repository
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null


# Install development tools and ROS tools
RUN apt-get update && apt-get install -y \
    python3-flake8-docstrings \
    python3-pip \
    python3-pytest-cov \
    # Replace 'ros-dev-tools' with actual ROS development tools package if available
    build-essential

# Install Python packages
RUN python3 -m pip install -U \
    flake8-blind-except \
    flake8-builtins \
    flake8-class-newline \
    flake8-comprehensions \
    flake8-deprecated \
    flake8-import-order \
    flake8-quotes \
    "pytest>=5.3" \
    pytest-repeat \
    pytest-rerunfailures

# Install vcstool
RUN apt-get update && apt-get install -y python3-vcstool


# Create a workspace and import ROS 2 code
RUN mkdir -p ~/ros2_humble/src && \
    cd ~/ros2_humble && \
    vcs import --input https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos src

# Upgrade all installed packages
RUN apt-get upgrade -y

#Install rosdep
RUN apt-get update && apt-get install -y python3-rosdep

RUN apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-overwrite" libopencv-dev opencv-libs opencv-licenses opencv-main opencv-python opencv-scripts


# Initialize rosdep and install dependencies
RUN cd ~/ros2_humble && \
    rosdep init && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers" --rosdistro humble -y

# Upgrade pip and install colcon
RUN python3 -m pip install --upgrade pip && \
    pip3 install -U colcon-common-extensions


# Build the ROS 2 workspace and source the setup script
RUN cd ~/ros2_humble && \
    colcon build --symlink-install --packages-up-to ros-base && \
    . install/local_setup.bash


    # Append the source command to .bashrc
RUN echo "source ~/ros2_humble/install/local_setup.bash" >> ~/.bashrc


RUN apt-get update





