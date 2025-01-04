FROM python:3.12-bullseye

# install uv to run stdio clients (uvx)
RUN pip install --no-cache-dir uv

# install npx to run stdio clients (npx)
RUN apt-get update && apt-get install -y --no-install-recommends curl
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN apt-get install -y --no-install-recommends nodejs
    
COPY pyproject.toml .

## FOR GHCR BUILD PIPELINE
COPY mcp_bridge/__init__.py mcp_bridge/__init__.py
COPY README.md README.md


RUN uv sync

COPY mcp_bridge mcp_bridge

EXPOSE 8000

WORKDIR /mcp_bridge
ENTRYPOINT ["uv", "run", "main.py"]
