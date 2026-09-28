"""
app/schemas/ingestion.py — Pydantic Schemas for Canonical Ingestion and Data Quality.
"""

from typing import Any
from pydantic import BaseModel, Field


class IngestionPreviewResponse(BaseModel):
    success: bool
    filename: str
    detected_headers: list[str]
    suggested_mapping: dict[str, str]
    total_rows: int
    preview_rows: list[dict[str, Any]]


class ColumnMappingSubmit(BaseModel):
    mapping: dict[str, str]


class DataQualityReportResponse(BaseModel):
    success: bool
    total_rows_scanned: int
    valid_rows_ingested: int
    quarantined_rows: int
    zero_sales_stockout_days: int
    negative_returns_clipped: int
    skus_profiled: int
    blocking_errors: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    active_dataset: str
