"""Download, preprocessing orchestration and quality control for CoRR NYU_2."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .registry import ROOT, Run

S3 = "https://fcp-indi.s3.amazonaws.com/"
PREFIX = "data/Projects/CORR/RawDataBIDS/NYU_2/"
RAW = ROOT / "data" / "external" / "raw" / "corr_nyu2"
DERIV = ROOT / "data" / "external" / "derived" / "corr_nyu2"
MIN_FREE_GB = 6.0
N_SUBSET = 4


def _get(url: str, tries: int = 6, headers: dict | None = None):
    for t in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=300)
        except Exception:
            time.sleep(5 * (t + 1))
    raise IOError(f"failed: {url}")


def listing() -> dict:
    cache = DERIV / "s3_listing.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out, token = {}, None
    while True:
        q = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"} | ({"continuation-token": token} if token else {})
        x = _get(S3 + "?" + urllib.parse.urlencode(q)).read().decode()
        for k, e, s in re.findall(r"<Key>([^<]+)</Key>.*?<ETag>&quot;([^&]+)&quot;</ETag><Size>(\d+)</Size>", x):
            out[k] = (int(s), e)
        m = re.search(r"<NextContinuationToken>([^<]+)", x)
        if not m:
            break
        token = m.group(1)
    DERIV.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def files_for(pid: str, keys: dict) -> dict:
    base = f"{PREFIX}sub-{pid}/ses-1/"
    f = {"run1": f"{base}func/sub-{pid}_ses-1_task-rest_run-1_bold.nii.gz",
         "run2": f"{base}func/sub-{pid}_ses-1_task-rest_run-2_bold.nii.gz",
         "t1": f"{base}anat/sub-{pid}_ses-1_run-1_T1w.nii.gz"}
    return f if all(v in keys for v in f.values()) else {}


def eligible(keys: dict) -> list[str]:
    ids = sorted({re.search(r"sub-(\d+)/", k).group(1) for k in keys if "/sub-" in k})
    return [p for p in ids if files_for(p, keys)]


def _free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1e9


ALLOW_METERED = False


def _metered() -> bool:
    import subprocess
    ps = ("[void][Windows.Networking.Connectivity.NetworkInformation, Windows, ContentType = WindowsRuntime]; "
          "$p = [Windows.Networking.Connectivity.NetworkInformation]::GetInternetConnectionProfile(); "
          "if ($p) { $p.GetConnectionCost().NetworkCostType } else { 'None' }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip()
    return out not in ("Unrestricted", "Unknown")


def _verify(path: Path, size: int, etag: str) -> dict:
    if path.stat().st_size != size:
        raise IOError(f"size mismatch {path.name}")
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        while (c := f.read(1 << 22)):
            md5.update(c)
            sha.update(c)
    if "-" not in etag and md5.hexdigest() != etag:
        raise IOError(f"MD5/ETag mismatch {path.name}")
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as g:
            while g.read(1 << 22):
                pass
    return {"bytes": size, "etag": etag, "md5_verified": "-" not in etag, "sha256": sha.hexdigest()}


def _fetch(k: str, dst: Path, size: int, chunk: int = 8 << 20, tries: int = 8) -> Path:
    tmp = dst.with_suffix(dst.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    if have > size:
        tmp.unlink()
        have = 0
    with open(tmp, "ab") as f:
        while have < size:
            hi = min(have + chunk, size) - 1
            for t in range(tries):
                try:
                    with _get(S3 + urllib.parse.quote(k), headers={"Range": f"bytes={have}-{hi}"}) as r:
                        data = r.read()
                    if len(data) == hi - have + 1:
                        break
                except Exception:
                    pass
                time.sleep(5 * (t + 1))
            else:
                raise IOError(f"ranged download failed: {k} at byte {have}")
            f.write(data)
            f.flush()
            have += len(data)
    return tmp


_MAN_LOCK = threading.Lock()
PARALLEL_FILES = 6


def _download_keys(ks: list[str], keys: dict, man_path: Path, parallel: int = PARALLEL_FILES):
    from concurrent.futures import ThreadPoolExecutor

    def one(k):
        dst = RAW / k[len(PREFIX):]
        if _free_gb() < MIN_FREE_GB + keys[k][0] / 1e9:
            raise SystemExit("storage guard reached during download")
        if not ALLOW_METERED and _metered():
            raise SystemExit("metered network connection: download paused (resume on an unmetered network or pass --allow-metered)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = _fetch(k, dst, keys[k][0])
        rec = _verify(tmp, *keys[k])
        tmp.replace(dst)
        with _MAN_LOCK:
            cur = json.loads(man_path.read_text()) if man_path.exists() else {}
            cur[k] = rec | {"time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            part = man_path.with_suffix(".json.tmp")
            part.write_text(json.dumps(cur, indent=0))
            part.replace(man_path)

    ex = ThreadPoolExecutor(max(1, parallel))
    try:
        for f in [ex.submit(one, k) for k in ks]:
            f.result()
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


def download(pids: list[str], keys: dict) -> list[dict]:
    man_path = DERIV / "download_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    top = PREFIX + "task-rest_bold.json"
    wanted = [top] + [k for p in pids for k in files_for(p, keys).values()]
    need = sum(keys[k][0] for k in wanted if k not in man) / 1e9
    if _free_gb() - need < MIN_FREE_GB:
        raise SystemExit(f"storage guard: {_free_gb():.1f} GB free, {need:.2f} GB needed, reserve {MIN_FREE_GB} GB")
    _download_keys([k for k in wanted if not (k in man and (RAW / k[len(PREFIX):]).exists())], keys, man_path)
    man = json.loads(man_path.read_text())
    return [man[k] | {"key": k} for k in wanted]


def fetch_sex(pids: list[str]) -> dict:
    import csv
    import io
    path = DERIV / "sex.json"
    sex = json.loads(path.read_text()) if path.exists() else {}
    for p in pids:
        if p in sex:
            continue
        t = _get(S3 + f"{PREFIX}sub-{p}/sub-{p}_sessions.tsv").read().decode()
        rows = [r for r in csv.DictReader(io.StringIO(t), delimiter="\t") if r.get("session_id") == "ses-1"]
        sex[p] = rows[0].get("sex") if rows else None
    path.write_text(json.dumps(sex, indent=0))
    return sex


def runlist(pids: list[str]) -> Path:
    wsl = lambda p: "/mnt/" + str(p).replace("\\", "/").replace(":", "", 1).lower()[0] + str(p).replace("\\", "/")[2:]
    lines = []
    for p in pids:
        d = RAW / f"sub-{p}" / "ses-1"
        for r in (1, 2):
            lines.append("|".join([f"sub-{p}_run{r}", wsl(d / "func" / f"sub-{p}_ses-1_task-rest_run-{r}_bold.nii.gz"),
                                   wsl(RAW / "task-rest_bold.json"), wsl(d / "anat" / f"sub-{p}_ses-1_run-1_T1w.nii.gz")]))
    out = DERIV / "runlist.txt"
    out.write_text("\n".join(lines) + "\n", newline="\n")
    return out


def _corr_masked(a, b, m):
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def _locations(pseudo: str, work: Path | None) -> dict:
    if work is not None and (work / "log.txt").exists():
        rp_dir = work / "RealignParameter" / pseudo
        return {"log": work / "log.txt", "trinfo": work / "TRInfo.tsv", "rp_dir": rp_dir, "qc": work / "QC" / pseudo,
                "roi": work / "Results" / "ROISignals_FunImgARCFW" / f"ROISignals_{pseudo}.mat",
                "aal": work / "Masks" / "aal_3mm_resliced.nii",
                "stages": {st: (work / f"P6_stage{st}.done").exists() for st in ("1a", "1b", "1c")},
                "slice_timing_dir": (work / "FunImgA" / pseudo).exists()}
    d = DERIV / pseudo
    flags = dict(l.split() for l in (d / "qc" / "stage_flags.txt").read_text().splitlines() if l.strip())
    return {"log": d / "dparsf_log.txt", "trinfo": d / "TRInfo.tsv", "rp_dir": d, "qc": d / "qc",
            "roi": d / f"ROISignals_{pseudo}.mat", "aal": d / "qc" / "aal_3mm_resliced.nii",
            "stages": {st: flags.get(f"stage{st}") == "1" for st in ("1a", "1b", "1c")},
            "slice_timing_dir": flags.get("slice_timing_dir") == "1"}


def qc_run(pseudo: str, work: Path | None, tpl_epi: Path) -> dict:
    import nibabel as nib
    import scipy.io as sio
    from scipy.signal import periodogram
    L = _locations(pseudo, work)
    r = {"pseudo": pseudo}
    log = L["log"].read_text(errors="ignore") if L["log"].exists() else ""
    r["stages_done"] = all(L["stages"].values()) and f"P6_ROI_OK {pseudo}" in log
    r["slice_timing_dir"] = L["slice_timing_dir"]
    ti = L["trinfo"].read_text().strip().splitlines() if L["trinfo"].exists() else []
    r["trinfo"] = ti[1] if len(ti) > 1 else None
    roi = sio.loadmat(L["roi"])["ROISignals"][:, :116].astype(float)
    r["T"], r["n_roi"] = roi.shape
    r["roi_finite"] = bool(np.isfinite(roi).all())
    r["roi_zero_var"] = int((roi.std(0) < 1e-8).sum())
    fd = np.loadtxt(next(L["rp_dir"].glob("FD_Jenkinson_*.txt")))
    rp = np.loadtxt(next(L["rp_dir"].glob("rp_*.txt")))
    r["fd_mean"], r["fd_max"] = float(fd.mean()), float(fd.max())
    r["rp_shape"] = list(rp.shape)
    r["max_abs_translation_mm"] = float(np.abs(rp[:, :3]).max())
    f, P = periodogram(roi - roi.mean(0), fs=0.5, window="hann", axis=0)
    tot = P[1:].sum(0)
    r["power_in_band_median"] = float(np.median(P[(f >= 0.01) & (f <= 0.08)].sum(0) / tot))
    r["power_above_0.1Hz_max"] = float((P[f > 0.1].sum(0) / tot).max())
    rpc = rp[-len(roi):] if len(rp) >= len(roi) else rp
    rc = np.corrcoef(np.column_stack([roi, rpc]).T)[:116, 116:]
    r["roi_motion_abs_corr_median"] = float(np.median(np.abs(rc)))
    q = L["qc"]
    wmean = nib.load(next(q.glob("wmean*.nii"))).get_fdata()
    tpl = nib.as_closest_canonical(nib.load(tpl_epi))
    from nibabel.processing import resample_from_to
    tpl_r = resample_from_to(tpl, nib.load(next(q.glob("wmean*.nii"))), order=1).get_fdata()
    m = (tpl_r > np.percentile(tpl_r[tpl_r > 0], 10)) & np.isfinite(wmean)
    r["meanEPI_vs_MNI_EPI_r"] = _corr_masked(np.nan_to_num(wmean), tpl_r, m)
    mask = nib.load(q / "brainmask_w.nii").get_fdata() > 0
    aal = nib.load(L["aal"]).get_fdata()
    labs = np.unique(aal[aal > 0])
    cov = np.array([mask[aal == l].mean() for l in labs])
    r["aal_labels"], r["aal_min_roi_coverage"], r["aal_rois_below_0.8"] = int(len(labs)), float(cov.min()), int((cov < 0.8).sum())
    r["pass"] = bool(r["stages_done"] and r["slice_timing_dir"] and r["T"] >= 150 and r["n_roi"] == 116 and r["roi_finite"]
                     and r["roi_zero_var"] == 0 and r["meanEPI_vs_MNI_EPI_r"] >= 0.6 and r["aal_labels"] == 116
                     and r["power_above_0.1Hz_max"] <= 0.02 and r["rp_shape"][1] == 6)
    return r


def _verified(qc: dict, p: str) -> bool:
    return all(qc.get(f"sub-{p}_run{r}", {}).get("pass") for r in (1, 2))


MEM_START_TWO_GB, MEM_CRIT_GB, MEM_RESUME_GB, PAGES_SEVERE = 1.5, 0.30, 0.5, 3000.0
RESLOG = Path(__import__("tempfile").gettempdir()) / "confirmation_resources.csv"


def _wsl(cmd: str) -> str:
    import subprocess
    r = subprocess.run(["wsl.exe", "-d", "P6Ubuntu", "--exec", "bash", "-c", cmd], capture_output=True)
    return r.stdout.decode(errors="ignore").replace("\x00", "")


def _host_sample() -> dict:
    import subprocess
    import psutil
    ps = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-Counter '\\Memory\\Pages/sec').CounterSamples[0].CookedValue"],
                        capture_output=True, text=True)
    try:
        pages = float(ps.stdout.strip().replace(",", "."))
    except ValueError:
        pages = float("nan")
    vhdx = ROOT / ".tools" / "wsl" / "ext4.vhdx"
    return {"t": time.strftime("%H:%M:%S"), "avail_gb": psutil.virtual_memory().available / 1e9, "pages_s": pages,
            "e_free_gb": _free_gb(), "vhdx_gb": vhdx.stat().st_size / 1e9 if vhdx.exists() else float("nan")}


BATT_START, BATT_STOP = 50, 15


def _on_battery_pct():
    import psutil
    b = psutil.sensors_battery()
    return None if (b is None or b.power_plugged) else b.percent


def _runner_count() -> int:
    out = _wsl("pgrep -fc '[b]ash -s /root/work/corr_nyu2/runlist.txt' || true").strip()
    return int(out.splitlines()[-1]) if out else 0


def _log_res(row: dict, n_workers: int, event: str = ""):
    new = not RESLOG.exists()
    with open(RESLOG, "a") as f:
        if new:
            f.write("t,avail_gb,pages_s,e_free_gb,vhdx_gb,workers,event\n")
        f.write(f"{row['t']},{row['avail_gb']:.2f},{row['pages_s']:.0f},{row['e_free_gb']:.2f},{row['vhdx_gb']:.2f},{n_workers},{event}\n")


def _prefetch(pids: list[str], keys: dict, also: list[str] = ()):
    try:
        man = json.loads((DERIV / "download_manifest.json").read_text())
        need = sum(keys[k][0] for p in list(pids) + list(also) for k in files_for(p, keys).values() if k not in man) / 1e9
        if pids and _free_gb() - need - 1.0 >= MIN_FREE_GB:
            download(pids, keys)
    except BaseException:
        pass


def full(keys: dict, el: list[str], batch: int, workers: int, dry: bool) -> dict:
    import subprocess
    qc_path = DERIV / "qc.json"
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}
    subset = DERIV / "qc_subset.json"
    if subset.exists():
        for q in json.loads(subset.read_text()):
            qc.setdefault(q["pseudo"], q | {"source": "subset"})
    todo = [p for p in el if not _verified(qc, p)]
    plan = {"eligible": len(el), "already_verified": len(el) - len(todo), "to_process": len(todo),
            "batches": (len(todo) + batch - 1) // batch, "batch": batch, "workers": workers, "free_gb": round(_free_gb(), 1)}
    if dry:
        return plan
    import os
    import psutil
    lock = DERIV / ".orchestrator.pid"
    old = int(lock.read_text() or 0) if lock.exists() else 0
    if old and psutil.pid_exists(old) and "corr_nyu2" in " ".join(psutil.Process(old).cmdline()):
        raise SystemExit(f"another orchestrator is running (pid {lock.read_text()}); not starting a duplicate")
    lock.write_text(str(os.getpid()))
    runner = ROOT / "src" / "p6" / "run_corr_dparsf.sh"
    tpl = ROOT / ".tools" / "subset_out" / "EPI.nii"
    reduced = False
    for i in range(0, len(todo), batch):
        b = todo[i:i + batch]
        pre = threading.Thread(target=_prefetch, args=(todo[i + batch:i + 2 * batch], keys, b), daemon=True)
        pre.start()
        while _runner_count() > 0:
            time.sleep(30)
        for lk in DERIV.glob(".lock_*"):
            lk.rmdir()
        h = _host_sample()
        t_wait = time.time()
        if h["avail_gb"] < MEM_CRIT_GB:
            while h["avail_gb"] < MEM_RESUME_GB:
                _log_res(h, 0, "pause_low_memory")
                if time.time() - t_wait > 3 * 3600:
                    raise SystemExit("host memory critically low for 3 h; stopped safely between batches")
                time.sleep(60)
                h = _host_sample()
        if (DERIV / ".stop").exists():
            raise SystemExit("stop file present; stopped safely between batches")
        t_b = time.time()
        while (pb := _on_battery_pct()) is not None and pb < BATT_START:
            _log_res(_host_sample(), 0, f"pause_battery_{pb}")
            if time.time() - t_b > 6 * 3600:
                raise SystemExit("on battery below 50 % for 6 h; stopped safely between batches")
            time.sleep(120)
        n_w = 2 if (workers >= 2 and not reduced and h["avail_gb"] >= MEM_START_TWO_GB) else 1
        man = json.loads((DERIV / "download_manifest.json").read_text())
        need_gb = sum(keys[k][0] for p in b for k in files_for(p, keys).values() if k not in man) / 1e9
        if _free_gb() - need_gb - 1.0 < MIN_FREE_GB:
            raise SystemExit(f"storage reserve: {_free_gb():.1f} GB free, batch needs {need_gb:.1f} GB; paused safely")
        download(b, keys)
        fetch_sex(b)
        runlist(b)
        subprocess.run(["wsl.exe", "-d", "P6Ubuntu", "--", "bash", "-c",
                        "mkdir -p /root/work/corr_nyu2 && cp '/path/to/project/data/external/derived/corr_nyu2/runlist.txt' /root/work/corr_nyu2/runlist.txt"],
                       check=True)
        _log_res(_host_sample(), n_w, f"batch_start_{i // batch + 1}_n{len(b)}")
        procs = [subprocess.Popen(["wsl.exe", "-d", "P6Ubuntu", "--", "bash", "-s", "/root/work/corr_nyu2/runlist.txt", str(w), str(n_w), "0"],
                                  stdin=open(runner, "rb"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for w in range(n_w)]
        low, pag, cut, batt_stop = 0, 0, False, False
        while any(pr.poll() is None for pr in procs):
            time.sleep(15)
            h = _host_sample()
            if (pb := _on_battery_pct()) is not None and pb <= BATT_STOP:
                _wsl("pkill -KILL -f '[b]ash -s /root/work/corr_nyu2/runlist.txt'; pkill -KILL -f '[o]ctave-cli'")
                _log_res(h, n_w, f"battery_stop_{pb}")
                batt_stop = True
                break
            low = low + 1 if h["avail_gb"] < MEM_CRIT_GB else 0
            pag = pag + 1 if h["pages_s"] > PAGES_SEVERE else 0
            ev = ""
            if h["e_free_gb"] < MIN_FREE_GB:
                _wsl("pkill -f '[b]ash -s /root/work/corr_nyu2/runlist.txt'; pkill -f '[o]ctave-cli'")
                _log_res(h, n_w, "storage_guard_stop")
                raise SystemExit("E: free space below the 6 GB reserve during processing; workers stopped")
            if n_w == 2 and (low >= 4 or pag >= 8):
                _wsl("d(){ for c in $(pgrep -P $1); do echo $c; d $c; done; }; "
                     "pid=$(pgrep -f '[b]ash -s /root/work/corr_nyu2/runlist.txt 1 2 0' | head -1); "
                     "[ -n \"$pid\" ] && kill -KILL $pid $(d $pid)")
                n_w, reduced, cut, ev = 1, True, True, "reduced_to_1_worker"
            _log_res(h, n_w, ev)
        pre.join()
        if cut and not batt_stop and any(not (DERIV / f"sub-{p}_run{r}" / f"ROISignals_sub-{p}_run{r}.mat").exists() for p in b for r in (1, 2)):
            while _runner_count() > 0:
                time.sleep(15)
            for lk in DERIV.glob(".lock_*"):
                lk.rmdir()
            subprocess.run(["wsl.exe", "-d", "P6Ubuntu", "--", "bash", "-s", "/root/work/corr_nyu2/runlist.txt", "0", "1", "0"],
                           stdin=open(runner, "rb"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        man = json.loads((DERIV / "download_manifest.json").read_text())
        for p in b:
            for r in (1, 2):
                s = f"sub-{p}_run{r}"
                try:
                    q = qc_run(s, None, tpl)
                    raws = [k for k in man if f"sub-{p}/" in k and (f"run-{r}_bold" in k or "T1w" in k)]
                    q["raw_sha256_ok"] = all(hashlib.sha256((RAW / k[len(PREFIX):]).read_bytes()).hexdigest() == man[k]["sha256"] for k in raws)
                    q["pairing_ok"] = True
                    q["derived_sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((DERIV / s).glob("*")) if f.is_file()}
                    q["pass"] = bool(q["pass"] and q["raw_sha256_ok"])
                except Exception as e:
                    q = {"pseudo": s, "pass": False, "error": repr(e)}
                qc[s] = q
            qc_path.write_text(json.dumps(qc, indent=0))
            if _verified(qc, p):
                shutil.rmtree(RAW / f"sub-{p}", ignore_errors=True)
                for r in (1, 2):
                    for f in (DERIV / f"sub-{p}_run{r}" / "qc").glob("*.nii"):
                        f.unlink()
                    qc[f"sub-{p}_run{r}"]["raw_deleted_after_verification"] = True
                qc_path.write_text(json.dumps(qc, indent=0))
        if batt_stop:
            raise SystemExit("battery at the 15 % guard: workers stopped, completed runs verified; resume on AC power")
        if _free_gb() < MIN_FREE_GB:
            raise SystemExit("storage guard after batch")
    lock.unlink()
    return plan | {"verified_now": sum(_verified(qc, p) for p in el)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "download", "qc", "full"])
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--workroot", default="//wsl.localhost/P6Ubuntu/root/work/corr_nyu2")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--allow-metered", action="store_true")
    a = ap.parse_args()
    global ALLOW_METERED
    ALLOW_METERED = a.allow_metered
    keys = listing()
    el = eligible(keys)
    pids = el[:N_SUBSET] if a.subset else el
    if a.step == "full":
        scope = el[: a.limit] if a.limit else el
        with Run("confirmation_preprocess_full" + ("_dry" if a.dry else ""), {"n": len(scope), "batch": a.batch, "workers": a.workers},
                 ["data/external/derived/corr_nyu2/qc.json"]):
            print(json.dumps(full(keys, scope, a.batch, a.workers, a.dry)))
        return
    if a.step == "select":
        print(len(el), "eligible; subset:", el[:N_SUBSET])
    elif a.step == "download":
        with Run("confirmation_download" + ("_subset" if a.subset else ""), {"n": len(pids), "ids": pids}, ["data/external/derived/corr_nyu2/download_manifest.json"]):
            recs = download(pids, keys)
            fetch_sex(pids)
            print(len(recs), "files verified;", f"{sum(r['bytes'] for r in recs) / 1e9:.2f} GB;", "free GB", round(_free_gb(), 1))
            print("runlist:", runlist(pids))
    else:
        tpl = ROOT / ".tools" / "subset_out" / "EPI.nii"
        man = json.loads((DERIV / "download_manifest.json").read_text())
        lines = [l.split("|") for l in (DERIV / "runlist.txt").read_text().splitlines() if l]
        res = []
        for p in pids:
            for r in (1, 2):
                s = f"sub-{p}_run{r}"
                q = qc_run(s, Path(a.workroot) / s, tpl)
                line = next(x for x in lines if x[0] == s)
                q["pairing_ok"] = (f"sub-{p}_ses-1_task-rest_run-{r}_bold" in line[1]) and (f"sub-{p}_ses-1_run-1_T1w" in line[3])
                raws = [k for k in man if f"sub-{p}/" in k and (f"run-{r}_bold" in k or "T1w" in k)]
                q["raw_sha256_ok"] = all(hashlib.sha256((RAW / k[len(PREFIX):]).read_bytes()).hexdigest() == man[k]["sha256"] for k in raws)
                q["derived_sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((DERIV / s).glob("*")) if f.is_file()}
                q["pass"] = bool(q["pass"] and q["pairing_ok"] and q["raw_sha256_ok"])
                res.append(q)
        out = DERIV / ("qc_subset.json" if a.subset else "qc.json")
        out.write_text(json.dumps(res, indent=1))
        for q in res:
            print(json.dumps({k: v for k, v in q.items() if k != "derived_sha256"}))


if __name__ == "__main__":
    main()
