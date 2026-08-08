FROM docker_ai_module:semantic-qa-phase1-perception

USER root

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    "transformers>=4.57,<5" \
    "accelerate>=1,<2" \
    "bitsandbytes>=0.46,<1" \
    "einops==0.8.2" \
    "roma==1.5.7" \
    "trimesh==5.0.0" \
    "scikit-learn==1.9.0"

# The base image is retained only for its CUDA/ROS/vision dependencies.  The
# retired scnav_vln implementation must not survive in the runnable image.
RUN rm -rf \
    /home/docker/ai_module/src/scnav_vln \
    /home/docker/ai_module/install/scnav_vln \
    /home/docker/ai_module/build/scnav_vln \
    /tmp/scnav_vln_logs

COPY --chown=docker:docker . /home/docker/ai_module
RUN chmod +x /home/docker/ai_module/docker/start_live_chain.sh && \
    mkdir -p /home/docker/ai_module/runs/live_robot && \
    chown -R docker:docker /home/docker/ai_module/runs

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/home/docker/ai_module

WORKDIR /home/docker/ai_module
USER docker
CMD ["/home/docker/ai_module/docker/start_live_chain.sh"]
