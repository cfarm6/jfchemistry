# Multi-stage build for jfchemistry package
# Build stage: Install dependencies using pixi with CUDA 13 support
FROM ghcr.io/prefix-dev/pixi:noble-cuda-13.0.0 AS build

WORKDIR /app

# Install git (required for JoltQC git dependency)
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Copy pixi configuration files and setup directory (needed for activation scripts)
COPY pyproject.toml pixi.lock ./
COPY setup/ ./setup/
# Copy LICENSE and README.md before pixi install (needed for package build)
COPY LICENSE README.md ./

# Install dependencies to `/app/.pixi/envs/default`
# Use `--locked` to ensure the lockfile is up to date with pixi.toml
RUN pixi install --locked && \
    # Remove the rattler cache to reduce image size
    rm -rf ~/.cache/rattler

# Create the shell-hook bash script to activate the environment
RUN pixi shell-hook -e default -s bash > /shell-hook

# Copy project source code
COPY jfchemistry/ ./jfchemistry/

# Create entrypoint script that activates the pixi environment
RUN echo "#!/bin/bash" > /app/entrypoint.sh && \
    cat /shell-hook >> /app/entrypoint.sh && \
    # Extend the shell-hook script to run the command passed to the container
    echo 'exec "$@"' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Production stage: Copy only the production environment
FROM nvidia/cuda:13.0.0-base-ubuntu24.04 AS production

WORKDIR /app

# Copy the production environment from build stage
# Note: The "prefix" (path) needs to stay the same as in the build container
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default

# Copy entrypoint script
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh

# Copy project code
COPY --from=build /app/jfchemistry /app/jfchemistry
COPY --from=build /app/setup /app/setup
COPY --from=build /app/README.md /app/README.md
COPY --from=build /app/LICENSE /app/LICENSE

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Use the entrypoint script to activate the pixi environment
ENTRYPOINT [ "/app/entrypoint.sh" ]

# Default command (can be overridden)
CMD [ "python", "-c", "import jfchemistry; print('JFChemistry is ready!')" ]
