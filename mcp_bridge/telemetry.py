from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from mcp_bridge.config import config

def setup_tracing(app) -> None:
    resource = Resource(attributes={"service.name": config.telemetry.service_name})

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    otlp_exporter = OTLPSpanExporter(endpoint=config.telemetry.otel_endpoint)
    span_processor = BatchSpanProcessor(otlp_exporter)

    if config.telemetry.enabled:
        # if not enabled do not add the span processor
        provider.add_span_processor(span_processor)

    FastAPIInstrumentor().instrument_app(app)
    HTTPXClientInstrumentor().instrument()