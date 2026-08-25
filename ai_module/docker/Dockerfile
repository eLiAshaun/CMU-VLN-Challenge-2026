FROM docker_ai_module:mast3r-live

# Thin runtime image for the task-conditioned online perception-memory-navigation chain.
# CUDA/ROS dependencies and checkpoint layers are inherited from the existing
# docker_ai_module:mast3r-live base; all business source is replaced below.

USER root

# Remove all stale business source code from base image before overlaying new code
RUN rm -rf \
    /home/docker/ai_module/integration \
    /home/docker/ai_module/integrations \
    /home/docker/ai_module/tools \
    /home/docker/ai_module/configs \
    /home/docker/ai_module/orchestration \
    /home/docker/ai_module/perception \
    /home/docker/ai_module/reasoning \
    /home/docker/ai_module/memory \
    /home/docker/ai_module/navigation \
    /home/docker/ai_module/config \
    /home/docker/ai_module/utils \
    /home/docker/ai_module/mast3r \
    /home/docker/ai_module/dust3r \
    /home/docker/ai_module/third_party \
    /home/docker/ai_module/docker \
    /home/docker/ai_module/vendor \
    /home/docker/ai_module/AGENTS.md \
    /home/docker/ai_module/README.md \
    /home/docker/ai_module/MODEL_ASSETS.md \
    /home/docker/ai_module/ARCHITECTURE_FINAL.md

# Overlay the current source over the inherited code.
COPY --chown=docker:docker integrations /home/docker/ai_module/integrations
COPY --chown=docker:docker tools /home/docker/ai_module/tools
COPY --chown=docker:docker configs /home/docker/ai_module/configs
COPY --chown=docker:docker orchestration /home/docker/ai_module/orchestration
COPY --chown=docker:docker navigation /home/docker/ai_module/navigation
COPY --chown=docker:docker config /home/docker/ai_module/config
COPY --chown=docker:docker utils /home/docker/ai_module/utils
COPY --chown=docker:docker mast3r /home/docker/ai_module/mast3r
COPY --chown=docker:docker dust3r /home/docker/ai_module/dust3r
COPY --chown=docker:docker third_party /home/docker/ai_module/third_party
COPY --chown=docker:docker docker /home/docker/ai_module/docker
COPY --chown=docker:docker vendor /home/docker/ai_module/vendor
COPY --chown=docker:docker LICENSE NOTICE requirements.txt /home/docker/ai_module/
COPY --chown=docker:docker AGENTS.md ARCHITECTURE_FINAL.md MODEL_ASSETS.md README.md /home/docker/ai_module/

RUN chmod +x /home/docker/ai_module/docker/start_live_chain.sh && \
    mkdir -p /home/docker/ai_module/runs/live_robot && \
    chown -R docker:docker /home/docker/ai_module/runs

USER docker
WORKDIR /home/docker/ai_module
CMD ["/home/docker/ai_module/docker/start_live_chain.sh"]
