from __future__ import annotations

from utils.shared_ui import _parse_trace_csv_bytes_list


class TestParseTraceCsvBytesList:
    def test_merges_and_dedupes_by_trace_id(self):
        csv1 = (
            "trace_id,timestamp,environment,session_id,user_id,latency_seconds,total_cost,prompt,answer\n"
            "t1,2025-01-01T00:00:00Z,production,s1,u1,1.0,0.1,hi,hello\n"
            "t2,2025-01-01T00:00:01Z,production,s2,u2,2.0,0.2,yo,there\n"
        ).encode("utf-8")
        csv2 = (
            "trace_id,timestamp,environment,session_id,user_id,latency_seconds,total_cost,prompt,answer\n"
            "t2,2025-01-01T00:00:01Z,production,s2,u2,2.0,0.2,yo,there\n"
            "t3,2025-01-01T00:00:02Z,staging,s3,u3,3.0,0.3,sup,ok\n"
        ).encode("utf-8")

        traces, stats = _parse_trace_csv_bytes_list([csv1, csv2])

        assert len(traces) == 3
        assert {t["id"] for t in traces} == {"t1", "t2", "t3"}
        assert stats["files"] == 2
        assert stats["rows"] == 4
        assert stats["loaded"] == 3
        assert stats["dupe_trace_id"] == 1

    def test_filters_machine_user_rows(self):
        csv1 = (
            "trace_id,timestamp,environment,session_id,user_id,latency_seconds,total_cost,prompt,answer\n"
            "t1,2025-01-01T00:00:00Z,production,s1,machine_user_123,1.0,0.1,hi,hello\n"
            "t2,2025-01-01T00:00:01Z,production,s2,u2,2.0,0.2,yo,there\n"
        ).encode("utf-8")

        traces, stats = _parse_trace_csv_bytes_list([csv1])

        assert len(traces) == 1
        assert traces[0]["id"] == "t2"
        assert stats["machine_user"] == 1
