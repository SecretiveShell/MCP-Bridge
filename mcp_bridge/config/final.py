from typing import Annotated, Literal, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field

from mcp.client.stdio import StdioServerParameters
from mcpx.client.transports.docker import DockerMCPServer


class InferenceServer(BaseModel):
    type: Literal["openai", "openrouter", "gemini"] = Field(
        "openai", description="Type of inference server"
    )  # used to apply data mappers
    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL of the inference server",
    )
    api_key: str = Field(
        default="unauthenticated", description="API key for the inference server"
    )


class Logging(BaseModel):
    log_level: Literal["INFO", "DEBUG"] = Field("INFO", description="default log level")
    log_server_pings: bool = Field(False, description="log server pings")


class SamplingModel(BaseModel):
    model: Annotated[str, Field(description="Name of the sampling model")]

    intelligence: Annotated[
        float, Field(description="Intelligence of the sampling model")
    ] = 0.5
    cost: Annotated[float, Field(description="Cost of the sampling model")] = 0.5
    speed: Annotated[float, Field(description="Speed of the sampling model")] = 0.5


class Sampling(BaseModel):
    timeout: Annotated[int, Field(description="Timeout for sampling requests")] = 10
    models: Annotated[
        list[SamplingModel], Field(description="List of sampling models")
    ] = []


class SSEMCPServer(BaseModel):
    # TODO: expand this once I find a good definition for this
    url: str = Field(description="URL of the MCP server")


MCPServer = Annotated[
    Union[StdioServerParameters, SSEMCPServer, DockerMCPServer],
    Field(description="MCP server configuration"),
]


class Network(BaseModel):
    host: str = Field("0.0.0.0", description="Host of the network")
    port: int = Field(8000, description="Port of the network")


class Settings(BaseSettings):
    inference_server: InferenceServer = Field(
        default_factory=lambda: InferenceServer.model_construct(),
        description="Inference server configuration",
    )

    mcp_servers: dict[str, MCPServer] = Field(
        default_factory=dict, description="MCP servers configuration"
    )

    sampling: Sampling = Field(
        default_factory=lambda: Sampling.model_construct(),
        description="sampling config",
    )

    logging: Logging = Field(
        default_factory=lambda: Logging.model_construct(),
        description="logging config",
    )

    network: Network = Field(
        default_factory=lambda: Network.model_construct(),
        description="network config",
    )

    model_config = SettingsConfigDict(
        env_prefix="MCP_BRIDGE__",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        cli_parse_args=True,
        cli_avoid_json=True,
    )
