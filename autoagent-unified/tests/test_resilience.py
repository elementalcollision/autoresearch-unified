"""Tests for autoresearch/resilience.py — atomic writes, TSV validation."""

import json
import os
import tempfile

import pytest

from autoresearch.resilience import atomic_write, atomic_append, validate_results_tsv, Heartbeat


class TestAtomicWrite:
    def test_creates_new_file(self, tmp_path):
        path = str(tmp_path / "test.txt")
        atomic_write(path, "hello world\n")
        assert open(path).read() == "hello world\n"

    def test_overwrites_existing_file(self, tmp_path):
        path = str(tmp_path / "test.txt")
        atomic_write(path, "first\n")
        atomic_write(path, "second\n")
        assert open(path).read() == "second\n"

    def test_no_tmp_file_left_behind(self, tmp_path):
        path = str(tmp_path / "test.txt")
        atomic_write(path, "content\n")
        assert not os.path.exists(path + ".tmp")


class TestAtomicAppend:
    def test_appends_to_file(self, tmp_path):
        path = str(tmp_path / "test.txt")
        with open(path, "w") as f:
            f.write("line1\n")
        atomic_append(path, "line2\n")
        assert open(path).read() == "line1\nline2\n"


class TestValidateResultsTsv:
    def test_valid_file(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        header = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\tbaseline_sha\twatts\tjoules_per_token\ttotal_energy_joules\n"
        data = "exp0\tbaseline\t1.500000\t8.0\t10000\t20.0\t100\tbaseline\tnone\tH100\tabc1234\t300.0\t0.001000\t500.0\n"
        with open(path, "w") as f:
            f.write(header + data)

        is_valid, warnings = validate_results_tsv(path)
        assert is_valid
        assert len(warnings) == 0

    def test_truncated_trailing_line(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        header = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\tbaseline_sha\twatts\tjoules_per_token\ttotal_energy_joules\n"
        truncated = "exp0\tbaseline\t1.5"  # No newline, incomplete fields
        with open(path, "w") as f:
            f.write(header + truncated)

        is_valid, warnings = validate_results_tsv(path)
        assert not is_valid
        assert any("Truncated" in w for w in warnings)

    def test_nonexistent_file(self):
        is_valid, warnings = validate_results_tsv("/nonexistent/path.tsv")
        assert is_valid
        assert len(warnings) == 0


class TestHeartbeat:
    def test_update_and_close(self, tmp_path):
        path = str(tmp_path / ".runner_status.json")
        hb = Heartbeat(path=path)
        hb.update(experiment=5, status="training", best_bpb=1.3)

        with open(path) as f:
            data = json.load(f)
        assert data["alive"] is True
        assert data["experiment"] == 5
        assert data["status"] == "training"

        hb.close()
        with open(path) as f:
            data = json.load(f)
        assert data["alive"] is False
        assert data["status"] == "stopped"
