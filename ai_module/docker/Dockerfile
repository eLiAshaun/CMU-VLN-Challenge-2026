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

RUN rm -rf \
    /home/docker/ai_module/src/scnav_vln \
    /home/docker/ai_module/install/scnav_vln \
    /home/docker/ai_module/build/scnav_vln \
    /tmp/scnav_vln_logs

COPY --chown=docker:docker . /home/docker/ai_module
RUN chmod +x /home/docker/ai_module/docker/start_live_chain.sh && \
    mkdir -p /home/docker/ai_module/runs/live_robot && \
    chown -R docker:docker /home/docker/ai_module/runs

# DeepSeek API key — set at image build time so it is baked into the
# submitted Docker image but not exposed in the public repository.
ARG DEEPSEEK_API_KEY=""
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/home/docker/ai_module \
    DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}

WORKDIR /home/docker/ai_module
USER docker
CMD ["/home/docker/ai_module/docker/start_live_chain.sh"]
