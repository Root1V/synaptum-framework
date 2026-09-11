"""Salida estructurada, sin atarse a una librería de modelos."""

from .derive import hints_of, json_schema_for, schema_of_type
from .protocol import DataclassSchema, PydanticSchema, RawSchema, Schema, schema_for

__all__ = [
    "Schema",
    "schema_for",
    "DataclassSchema",
    "PydanticSchema",
    "RawSchema",
    "json_schema_for",
    "schema_of_type",
    "hints_of",
]
