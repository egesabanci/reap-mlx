# Base image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Prevents prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including git, sudo, and add deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    software-properties-common \
    sudo \
    ssh \
    tmux \
    vim \
    htop \
    unzip \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv system-wide and create python symlink
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    ln -s /usr/bin/python3.12 /usr/bin/python

# Add build arguments for user and group IDs
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create a non-root user 'dev' and add to sudo group
RUN groupadd -g $GROUP_ID dev && \
    useradd -u $USER_ID -g $GROUP_ID -s /bin/bash -m dev && \
    adduser dev sudo && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R dev:dev /home/dev

# Set up the working directory
WORKDIR /app

# Copy project files
COPY . .

RUN chown -R dev:dev /app

# Create a cache directory for uv inside /app and set ownership
RUN mkdir -p /app/.uv_cache && chown -R dev:dev /app/.uv_cache

# Set UV_CACHE_DIR environment variable
ENV UV_CACHE_DIR="/app/.uv_cache"

# Switch to the new user
USER dev

# Set the entrypoint to run our setup script via bash
# This avoids host filesystem permission issues with the script itself.
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]

# Add venv to PATH for interactive shells
ENV PATH="/app/.venv/bin:/home/dev/.local/bin:${PATH}"

# Default command to run if no other command is specified
CMD ["/bin/bash"]