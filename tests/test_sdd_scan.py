"""SDD 標註掃描的測試：用自己造的小標註檔驗證「最長連續可見段」與篩選規則。"""
import pytest

from demo.sdd_scan import parse_annotations, scan, track_stats


def write_ann(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(" ".join(str(v) for v in r) for r in rows) + "\n", encoding="utf-8")


def moving(tid, label, frames, lost_at=(), step=2):
    return [(tid, 100 + step * f, 100, 120 + step * f, 120, f, int(f in lost_at), 0, 0, f'"{label}"') for f in frames]


def test_parse_and_longest_visible_segment(tmp_path):
    f = tmp_path / "a.txt"
    write_ann(f, moving(3, "Car", range(0, 100), lost_at=range(40, 50)))
    tracks = parse_annotations(f)
    st = track_stats("v", 3, tracks[3])
    assert st.label == "Car"
    assert st.longest_visible_s == pytest.approx(50 / 30)  # frame 50–99 共 50 幀
    assert st.path_length_px == pytest.approx(49 * 2)
    assert st.mean_box_px == pytest.approx(20)


def test_scan_filters_by_label_duration_and_motion(tmp_path):
    root = tmp_path / "sdd"
    write_ann(root / "annotations/nexus/video0/annotations.txt",
              moving(1, "Car", range(0, 1200)) + moving(2, "Pedestrian", range(0, 1200)) + moving(3, "Car", range(0, 1200), step=0))
    write_ann(root / "annotations/deathCircle/video1/annotations.txt", moving(7, "Biker", range(0, 1000)))
    found = scan(root, min_seconds=30.0, min_path_px=300.0)
    assert [(s.video, s.track_id) for s in found] == [("nexus/video0", 1), ("deathCircle/video1", 7)]  # 車輛優先；停著的車與行人被排除
