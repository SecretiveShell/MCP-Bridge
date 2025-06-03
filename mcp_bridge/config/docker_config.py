from pydantic import BaseModel, Field

class DockerMCPServer(BaseModel):
    container_name: str | None = Field(default=None, description="Name of the docker container")
    image: str = Field(description="Image of the docker container")
    args: list[str] = Field(default_factory=list, description="Command line arguments for the docker container")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables for the docker container")