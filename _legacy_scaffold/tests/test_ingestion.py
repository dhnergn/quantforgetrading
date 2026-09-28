"""Unit tests for data ingestion."""
import pandas as pd
import pytest

from quantforge.data.ingestion import DataIngestion, OHLCVSchema


class TestDataIngestion:
    def test_ingest_dataframe(self):
        """Should ingest a DataFrame correctly."""
        df = pd.DataFrame({
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [95.0, 96.0],
            "close": [102.0, 103.0],
            "volume": [1000, 2000],
        }, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))

        ingestion = DataIngestion()
        result = ingestion.ingest_dataframe(df, "TEST")

        assert "instrument_id" in result.columns
        assert "source" in result.columns
        assert "ingestion_timestamp" in result.columns
        assert result["instrument_id"].iloc[0] == "TEST"

    def test_ingestion_log(self):
        """Should log ingestion events."""
        df = pd.DataFrame({
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "volume": [1000],
        }, index=pd.to_datetime(["2023-01-01"]))

        ingestion = DataIngestion()
        ingestion.ingest_dataframe(df, "TEST")

        log = ingestion.get_ingestion_log()
        assert len(log) == 1
        assert log[0]["instrument_id"] == "TEST"
        assert log[0]["rows_ingested"] == 1
