"""งาน 9 ข้อ 1–5: ระบบรันจริง — เขตเวลา/นาฬิกา, สถานะงานเย็น, แจ้งเตือน, ตรวจความพร้อม"""
import importlib.util
import json
from datetime import timedelta

import pandas as pd
import pytest

import forecast.live.evening as EV
import forecast.live.ready as RD
import forecast.runtime as RT
from forecast.config import ROOT, load_rules
from forecast.live.notify import notify

RULES = load_rules()


# ---------- ข้อ 1–2: เขตเวลา / นาฬิกา / log ตอนเริ่ม
def test_timezone_not_bangkok_stops(monkeypatch):
    monkeypatch.setattr(RT, "local_offset", lambda: timedelta(0))
    monkeypatch.delattr(RT.time, "tzset", raising=False)
    with pytest.raises(SystemExit, match="Asia/Bangkok"):
        RT.check_timezone()


def test_timezone_bangkok_ok(monkeypatch):
    monkeypatch.setattr(RT, "local_offset", lambda: timedelta(hours=7))
    assert RT.check_timezone() is None


@pytest.mark.parametrize("skew, warn", [(120.0, True), (-75.0, True), (5.0, False)])
def test_startup_logs_and_warns_on_clock_skew(monkeypatch, tmp_path, capsys, skew, warn):
    monkeypatch.setattr(RT, "local_offset", lambda: timedelta(hours=7))
    monkeypatch.setattr(RT, "clock_skew", lambda: (skew, "https://ref"))
    rec = RT.startup("test_script", log=tmp_path / "run_start.jsonl")
    out = capsys.readouterr().out
    assert rec["clock_warning"] is warn and ("นาฬิกาเครื่องต่างจากเวลาอ้างอิง" in out) is warn
    logged = json.loads((tmp_path / "run_start.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    for k in ["script", "started_at", "utc_offset", "commit", "host", "network", "clock_skew_sec"]:
        assert k in logged
    assert logged["utc_offset"] == "+07:00" and "test_script" in out and logged["host"] in out


def test_clock_unreachable_does_not_stop(monkeypatch, tmp_path):
    monkeypatch.setattr(RT, "local_offset", lambda: timedelta(hours=7))
    monkeypatch.setattr(RT, "reference_time", lambda *a, **k: (None, "ตรวจนาฬิกาไม่ได้ (URLError)"))
    rec = RT.startup("x", log=tmp_path / "r.jsonl")
    assert rec["clock_skew_sec"] is None and not rec["clock_warning"]


# ---------- ข้อ 3: สถานะงานเย็น
def _fake_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(EV, "OUTPUT_DIR", tmp_path)


def test_evening_status_and_rerun_decision(monkeypatch, tmp_path):
    _fake_outputs(monkeypatch, tmp_path)
    day = pd.Timestamp("2026-11-09")
    assert EV.needs_rerun(day)[0] and "ไม่มี" in EV.needs_rerun(day)[1]
    st, _ = EV.write_status(day, {"settrade 1d": True, "yfinance": True})
    assert not st["ok"] and EV.needs_rerun(day)[0] and "draft" in EV.needs_rerun(day)[1]
    (tmp_path / "predictions" / "draft").mkdir(parents=True)
    (tmp_path / "predictions" / "draft" / "2026-11-09_morning.jsonl").write_text(
        json.dumps({"model_version": "E12-fd_lr-test"}) + "\n", encoding="utf-8")
    (tmp_path / "models" / "stage2").mkdir(parents=True)
    (tmp_path / "models" / "stage2" / "2026-11-09.json").write_text(json.dumps({"reason": None}), encoding="utf-8")
    st, _ = EV.write_status(day, {"settrade 1d": True, "yfinance": True})
    assert st["ok"] and st["draft"]["model_version"] == "E12-fd_lr-test" and EV.needs_rerun(day) == (False, None)
    st, _ = EV.write_status(day, {"settrade 1d": False, "yfinance": True})
    assert EV.needs_rerun(day)[0] and "settrade 1d" in EV.needs_rerun(day)[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_morning_reruns_evening_job_when_status_bad(monkeypatch, tmp_path):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.setattr(RT, "clock_skew", lambda: (0.0, "test"))
    monkeypatch.setattr(sys, "argv", ["05_predict_live.py"])
    mod = _load_script("05_predict_live")
    calls = []
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    monkeypatch.setattr(mod, "needs_rerun", lambda d: (True, "ไม่มี evening_status"))
    mod.ensure_evening(pd.Timestamp("2026-11-09"))
    assert calls and "00_collect_daily.py" in calls[0][1] and "--target-date" in calls[0] and "2026-11-09" in calls[0]
    calls.clear()
    monkeypatch.setattr(mod, "needs_rerun", lambda d: (False, None))
    mod.ensure_evening(pd.Timestamp("2026-11-09"))
    assert not calls


# ---------- ข้อ 4: แจ้งเตือน
SECRET = "https://discord.com/api/webhooks/123/SECRET-TOKEN"


def test_notify_failure_never_raises_and_hides_secret(tmp_path):
    def boom(text):
        raise RuntimeError(f"cannot reach {SECRET}")
    sent = []
    res = notify("ready", "ทดสอบ", ["KBANK ขึ้น"], outbox=tmp_path / "o.jsonl",
                 chans={"discord": boom, "slack": lambda t: sent.append(t)})
    assert res == {"discord": "RuntimeError", "slack": "ok"} and "KBANK ขึ้น" in sent[0]
    text = (tmp_path / "o.jsonl").read_text(encoding="utf-8")
    assert "SECRET-TOKEN" not in text and "ทดสอบ" in text


def test_notify_survives_broken_outbox(tmp_path):
    bad = tmp_path / "dir_not_file"
    bad.mkdir()
    assert notify("failed", "x", outbox=bad, chans={}) == {}


def test_notify_unreachable_channel_returns_error_type(tmp_path):
    from forecast.live.notify import channels
    ch = channels({"NOTIFY_SLACK_WEBHOOK": "http://127.0.0.1:9/hook"})
    res = notify("info", "x", outbox=tmp_path / "o.jsonl", chans=ch)
    assert res["slack"] in ("URLError", "ConnectionRefusedError", "TimeoutError") or res["slack"].startswith("URLError")
    assert "127.0.0.1" not in (tmp_path / "o.jsonl").read_text(encoding="utf-8")


# ---------- ข้อ 5: ตรวจความพร้อม
def test_targets():
    assert RD.targets(RULES, "2026-11-08 20:00") == (pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-09"))
    assert RD.targets(RULES, "2026-11-09 09:00") == (pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-09"))
    assert RD.targets(RULES, "2026-11-09 10:00") == (pd.Timestamp("2026-11-10"), pd.Timestamp("2026-11-09"))
    assert RD.targets(RULES, "2026-11-13 18:00") == (pd.Timestamp("2026-11-16"), pd.Timestamp("2026-11-16"))


def test_check_ready_flags_missing_models(monkeypatch, tmp_path):
    monkeypatch.setattr(RD, "clock_skew", lambda: (2.0, "ref"))
    monkeypatch.setattr(RD, "OUTPUT_DIR", tmp_path)
    t, _, _ = RD.run_checks(RULES, now="2026-11-01 18:00", settrade=False)
    models = t[t["check"].str.startswith("โมเดล")]
    assert len(models) == 5 and (models["status"] == "FAIL").all()
    assert set(t["check"]) >= {"เขตเวลา", "นาฬิกา", ".env Settrade", ".env แจ้งเตือน Telegram"}
    assert t.loc[t["check"] == "นาฬิกา", "status"].iat[0] == "PASS"
    assert "SECRET" not in t.to_string()


def test_check_ready_clock_fail(monkeypatch):
    monkeypatch.setattr(RD, "clock_skew", lambda: (95.0, "ref"))
    t = pd.DataFrame(RD.check_clock_tz(), columns=["check", "status", "detail"])
    assert t.loc[t["check"] == "นาฬิกา", "status"].iat[0] == "FAIL"


def test_every_message_starts_with_host(tmp_path):
    import platform
    sent = []
    notify("ready", "ผลทำนาย", ["KBANK ...", "ADVANC ..."], outbox=tmp_path / "o.jsonl", chans={"t": sent.append})
    assert sent[0].startswith(f"[{platform.node()}] ")


class _FakeMarket:
    """API จำลอง: วันก่อน `oldest` โยน error (แบบ VM 2 พ.ย.), วันอื่นคืน 2 แท่ง"""
    def __init__(self, oldest, fail_all=False):
        self.oldest, self.fail_all = pd.Timestamp(oldest), fail_all

    def get_candlestick(self, symbol, interval, limit, start, end, normalized):
        d = pd.Timestamp(start[:10])
        if self.fail_all or d < self.oldest:
            raise RuntimeError("range not available")
        t = [(d + pd.Timedelta(hours=3, minutes=m)).tz_localize("Asia/Bangkok").timestamp() for m in (0, 1)]
        return {"time": t, "open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [1, 1], "volume": [1, 1]}


def test_fetch_days_skips_old_day_errors():
    from forecast.data.settrade import failed_days, fetch_days
    rules, dates = load_rules(), pd.bdate_range("2026-10-26", "2026-11-02")
    frames, empty, errors = fetch_days(_FakeMarket("2026-10-29"), "KBANK", dates, rules)
    assert [d.date().isoformat() for d, _ in errors] == ["2026-10-26", "2026-10-27", "2026-10-28"]
    assert len(frames) == 3 and empty == 3
    assert not failed_days(dates, errors)                 # error เฉพาะวันเก่า = ปกติ


def test_daily_bar_kept_after_close_not_midnight():
    from forecast.data.io import drop_unfinished
    rules = load_rules()
    d = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-30", "2026-11-02"])})
    kept = lambda now: drop_unfinished(d, rules, pd.Timestamp(now), 1440)["timestamp"].dt.date.astype(str).tolist()
    assert kept("2026-11-02 16:30") == ["2026-10-30"]                 # ATC ยังไม่จบ → ยังไม่เก็บวันนี้
    assert kept("2026-11-02 17:30") == ["2026-10-30", "2026-11-02"]   # งานเย็นเก็บวันนี้ได้
    m = pd.DataFrame({"timestamp": pd.to_datetime(["2026-11-02 12:14", "2026-11-02 12:15"])})
    assert len(drop_unfinished(m, rules, pd.Timestamp("2026-11-02 12:15"), 1)) == 1   # 1m เหมือนเดิม


def test_fetch_days_fails_when_latest_or_all_fail():
    from forecast.data.settrade import failed_days, fetch_days
    rules, dates = load_rules(), pd.bdate_range("2026-10-29", "2026-11-02")
    _, _, errors = fetch_days(_FakeMarket("2026-11-03"), "KBANK", dates, rules)   # วันล่าสุด error
    assert failed_days(dates, errors)
    _, _, errors = fetch_days(_FakeMarket("2026-10-01", fail_all=True), "KBANK", dates, rules)
    assert failed_days(dates, errors)
