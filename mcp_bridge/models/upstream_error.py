from pydantic import BaseModel, Field
from typing import Annotated

    
class UpstreamErrorDetails(BaseModel):
    message: Annotated[str, Field(description="Error message")] = "An upstream error occurred"
    code : Annotated[str | None, Field(description="Error code")] = "UPSTREAM_ERROR"

class UpstreamError(BaseModel):
    error: Annotated[UpstreamErrorDetails, Field(description="Error details")]