
variable "RELEASE" {
    default = "v1.0.0"
}

variable "REGISTRY" {
    default = "ghcr.io/secretiveshell/mcp-bridge"
}

group "default" {
  targets = ["mcp-bridge"]
}


target "mcp-bridge" {
  dockerfile = "Dockerfile"
  tags       = ["${REGISTRY}/${target.mcp-bridge.name}:${RELEASE}"]
  context    = "."
  labels = {
    "org.opencontainers.image.source" = "https://github.com/SecretiveShell/MCP-Bridge"
  }
}
