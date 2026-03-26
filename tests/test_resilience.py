"""Tests for tui.resilience -- crash-safe file operations and validation."""
import json
import os
import tempfile
import pytest
from tui.resilience import atomic_write, atomic_append, validate_results_tsv, Heartbeat


class TestAtomicWrite:
    """Test crash-safe file writing."""

    def test_basic_write(self, tmp_path):
        path = str(tmp_path / "test.txt")
        atomic_write(path, "hello world")
        assert open(path).read() == "hello world"

    def test_overwrite(self, tmp_path):
        path = str(tmp_path / "test.txt")
        atomic_write(path, "first")
        atomic_write(path, "second")
        assert open(path).read() == "second"

    def test_no_tmp_file_left(self, tmp_path):
        path = str(tmp_path / "test.txt")
        atomic_write(path, "data")
        assert not os.path.exists(path + ".tmp")

    def test_empty_content(self, tmp_path):
        path = str(tmp_path / "test.txt")
        atomic_write(path, "")
        assert open(path).read() == ""


class TestAtomicAppend:
    """Test append with fsync."""

    def test_append_to_new_file(self, tmp_path):
        path = str(tmp_path / "test.txt")
        atomic_append(path, "line1\n")
        assert open(path).read() == "line1\n"

    def test_append_multiple_lines(self, tmp_path):
        path = str(tmp_path / "test.txt")
        atomic_append(path, "line1\n")
        atomic_append(path, "line2\n")
        assert open(path).read() == "line1\nline2\n"

    def test_append_preserves_existing(self, tmp_path):
        path = str(tmp_path / "test.txt")
        with open(path, "w") as f:
            f.write("existing\n")
        atomic_append(path, "new\n")
        assert open(path).read() == "existing\nnew\n"


class TestValidateResultsTsv:
    """Test TSV validation and corruption recovery."""

    HEADER = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\tbaseline_sha\n"
    VALID_ROW = "1\tbaseline\t1.234\t6.0\t100000\t5.0\t500\tkeep\t\tRTX 5070 Ti\tabc1234\n"

    HEADER_14 = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\tbaseline_sha\twatts\tjoules_per_token\ttotal_energy_joules\n"
    VALID_ROW_14 = "1\tbaseline\t1.234\t6.0\t100000\t5.0\t500\tkeep\t\tRTX 5070 Ti\tabc1234\t350.5\t0.001378\t105210.5\n"

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        with open(path, "w") as f:
            f.write("")
        valid, warnings = validate_results_tsv(path)
        assert valid is True

    def test_missing_file(self, tmp_path):
        valid, warnings = validate_results_tsv(str(tmp_path / "nonexistent.tsv"))
        assert valid is True

    def test_valid_file(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        with open(path, "w") as f:
            f.write(self.HEADER + self.VALID_ROW)
        valid, warnings = validate_results_tsv(path)
        assert valid is True
        assert len(warnings) == 0

    def test_truncated_line_removed(self, tmp_path):
        """Simulate a crash mid-write -- partial line with no trailing newline."""
        path = str(tmp_path / "results.tsv")
        with open(path, "w") as f:
            f.write(self.HEADER + self.VALID_ROW)
            f.write("2\tincomplete")  # No newline, too few fields
        valid, warnings = validate_results_tsv(path)
        assert any("truncated" in w.lower() for w in warnings)
        # After fix, the truncated line should be gone
        content = open(path).read()
        assert "incomplete" not in content

    def test_valid_14col_file(self, tmp_path):
        """14-column files (with energy data) validate cleanly."""
        path = str(tmp_path / "results.tsv")
        with open(path, "w") as f:
            f.write(self.HEADER_14 + self.VALID_ROW_14)
        valid, warnings = validate_results_tsv(path)
        assert valid is True
        assert len(warnings) == 0

    def test_legacy_11col_still_valid(self, tmp_path):
        """11-column files (pre-energy) must still validate as OK."""
        path = str(tmp_path / "results.tsv")
        with open(path, "w") as f:
            f.write(self.HEADER + self.VALID_ROW)
        valid, warnings = validate_results_tsv(path)
        assert valid is True
        assert len(warnings) == 0

    def test_wrong_column_count_rejected(self, tmp_path):
        """12-column rows are not valid (neither legacy nor current)."""
        path = str(tmp_path / "results.tsv")
        with open(path, "w") as f:
            f.write(self.HEADER_14)
            f.write("1\tbaseline\t1.234\t6.0\t100000\t5.0\t500\tkeep\t\tRTX\tabc\t350\n")  # 12 cols
        valid, warnings = validate_results_tsv(path)
        assert not valid
        assert any("expected" in w.lower() for w in warnings)


class TestHeartbeat:
    """Test heartbeat status file."""

    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "status.json")
        hb = Heartbeat(path)
        hb.update(experiment=1, status="training")
        assert os.path.exists(path)

    def test_json_valid(self, tmp_path):
        path = str(tmp_path / "status.json")
        hb = Heartbeat(path)
        hb.update(experiment=5, status="training", dataset="pubmed", best_bpb=0.95)
        data = json.loads(open(path).read())
        assert data["experiment"] == 5
        assert data["status"] == "training"
        assert data["alive"] is True

    def test_close_marks_not_alive(self, tmp_path):
        path = str(tmp_path / "status.json")
        hb = Heartbeat(path)
        hb.update(experiment=1, status="running")
        hb.close()
        data = json.loads(open(path).read())
        assert data["alive"] is False
        assert data["status"] == "stopped"

    def test_update_overwrites(self, tmp_path):
        path = str(tmp_path / "status.json")
        hb = Heartbeat(path)
        hb.update(experiment=1, status="training")
        hb.update(experiment=2, status="evaluating")
        data = json.loads(open(path).read())
        assert data["experiment"] == 2
