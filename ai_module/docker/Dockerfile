FROM elias1012/cmu-vln-2026:latest

USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3.12-venv && \
    python3 -m venv --system-site-packages /opt/cmu-rebuild-venv
ENV PATH=/opt/cmu-rebuild-venv/bin:$PATH
COPY rebuild/requirements.txt /tmp/cmu-rebuild-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/cmu-rebuild-requirements.txt
RUN git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git /opt/cmu-da3 && \
    python3 -m pip install --no-cache-dir --no-deps /opt/cmu-da3

# The base contains an old AI checkout. Remove it from the derived filesystem
# before copying the selected implementation; base layers/host assets stay intact.
RUN rm -r -- /home/docker/ai_module
COPY --chown=docker:docker checkpoints/rebuild /home/docker/ai_module/checkpoints/rebuild

COPY rebuild/requirements-compiler.txt /tmp/cmu-rebuild-compiler-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/cmu-rebuild-compiler-requirements.txt
COPY --chown=docker:docker rebuild /home/docker/ai_module/rebuild
COPY --chown=docker:docker configs/rebuild.json configs/model_assets.json configs/sensor_at_scan_to_panorama.json /home/docker/ai_module/configs/
COPY --chown=docker:docker docker/start_live_chain.sh /home/docker/ai_module/docker/start_live_chain.sh
COPY --chown=docker:docker ARCHITECTURE_FINAL.md MODEL_ASSETS.md README.md LICENSE NOTICE CHECKPOINTS_NOTICE /home/docker/ai_module/
RUN chmod +x /home/docker/ai_module/docker/start_live_chain.sh && \
    chmod o+x /home/docker /home/docker/ai_module && \
    mkdir -p /home/docker/ai_module/runs/rebuild /tmp/cmu-rebuild-hf /tmp/cmu-rebuild-torch-kernels && \
    chmod 1777 /tmp/cmu-rebuild-hf /tmp/cmu-rebuild-torch-kernels && \
    chown -R docker:docker /home/docker/ai_module/runs/rebuild
ENV PYTHONPATH=/home/docker/ai_module HF_HOME=/tmp/cmu-rebuild-hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/cmu-rebuild-inductor TRITON_CACHE_DIR=/tmp/cmu-rebuild-triton \
    PYTORCH_KERNEL_CACHE_PATH=/tmp/cmu-rebuild-torch-kernels \
    ROS_LOG_DIR=/tmp/cmu-rebuild-ros MPLCONFIGDIR=/tmp/cmu-rebuild-matplotlib \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTDDS_BUILTIN_TRANSPORTS=UDPv4
USER docker
WORKDIR /home/docker/ai_module
CMD ["/home/docker/ai_module/docker/start_live_chain.sh"]
