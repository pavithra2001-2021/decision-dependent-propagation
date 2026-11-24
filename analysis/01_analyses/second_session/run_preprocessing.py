"""Second session: preprocessing and between- and within-session change."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from p6 import corr_nyu2 as CN

RAWDIR, DERIV, LOG = HERE / "work" / "raw", HERE / "derived", HERE / "pipeline_progress.txt"
MIN_FREE_GB = 4.0
PREFIX = CN.PREFIX
WSL_OUT = "/path/to/project/analysis/01_analyses/second_session/derived"
KEYS = json.loads((ROOT / "data/external/derived/corr_nyu2/s3_listing.json").read_text())


def wslpath(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:]


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1e9


def names(p):
    return [f"sub-{p}_ses2_run{r}" for r in (1, 2)]


def done(p):
    return all((DERIV / n / f"ROISignals_{n}.mat").exists() for n in names(p))


def download(p):
    t0 = time.time()
    d = RAWDIR / p
    ks = {"run1": f"{PREFIX}sub-{p}/ses-2/func/sub-{p}_ses-2_task-rest_run-1_bold.nii.gz",
          "run2": f"{PREFIX}sub-{p}/ses-2/func/sub-{p}_ses-2_task-rest_run-2_bold.nii.gz",
          "t1": f"{PREFIX}sub-{p}/ses-1/anat/sub-{p}_ses-1_run-1_T1w.nii.gz"}
    local, recs = {}, {}
    for nm, k in ks.items():
        dst = d / Path(k).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            size, etag = KEYS[k]
            tmp = CN._fetch(k, dst, size)
            recs[k] = CN._verify(tmp, size, etag) | {"time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            tmp.replace(dst)
        local[nm] = dst
    return local, recs, time.time() - t0


def main():
    pids = [p.strip() for p in (HERE / "ses2_candidates.txt").read_text().split() if p.strip()]
    todo = [p for p in pids if not done(p)]
    side = RAWDIR / "task-rest_bold.json"
    driver = (ROOT / "src/p6/run_corr_dparsf.sh").read_text(encoding="utf-8")
    script = f"export P6_CORR_OUT='{WSL_OUT}'\n" + driver
    man_path = HERE / "download_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    log(f"start v3 (prefetch + 2 workers): {len(todo)} participants to do; free {free_gb():.2f} GB")
    ex = ThreadPoolExecutor(1)
    fut = ex.submit(download, todo[0]) if todo else None
    for i, p in enumerate(todo):
        if free_gb() < MIN_FREE_GB:
            log(f"STOP disk guard: {free_gb():.2f} GB free")
            break
        try:
            local, recs, tdl = fut.result()
        except Exception as e:
            log(f"{p} DOWNLOAD FAILED {e}")
            shutil.rmtree(RAWDIR / p, ignore_errors=True)
            fut = ex.submit(download, todo[i + 1]) if i + 1 < len(todo) else None
            continue
        manifest |= recs
        man_path.write_text(json.dumps(manifest, indent=0))
        fut = ex.submit(download, todo[i + 1]) if i + 1 < len(todo) else None
        t1 = time.time()
        d = RAWDIR / p
        rl = "\n".join(f"{n}|{wslpath(local[f'run{r}'])}|{wslpath(side)}|{wslpath(local['t1'])}" for r, n in zip((1, 2), names(p))) + "\n"
        (d / "runlist.txt").write_text(rl, encoding="utf-8", newline="\n")
        subprocess.run(["wsl.exe", "-d", "P6Ubuntu", "-u", "root", "--", "bash", "-c",
                        f"mkdir -p /root/work/ses2 && cp '{wslpath(d / 'runlist.txt')}' /root/work/ses2/runlist_{p}.txt"], check=True)
        procs = [subprocess.Popen(["wsl.exe", "-d", "P6Ubuntu", "-u", "root", "--", "bash", "-s", f"/root/work/ses2/runlist_{p}.txt", str(w), "2", "0"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) for w in (1, 2)]
        for pr in procs:
            pr.stdin.write(script.encode())
            pr.stdin.close()
        outs = [pr.stdout.read().decode(errors="replace").strip() for pr in procs]
        for pr in procs:
            pr.wait(timeout=7200)
        ok = done(p)
        log(f"{p} {'OK' if ok else 'FAILED'} download {tdl:.0f}s preprocess {time.time() - t1:.0f}s free {free_gb():.2f} GB "
            f"[{i + 1}/{len(todo)}] {' | '.join(o[-30:] for o in outs)}")
        shutil.rmtree(d, ignore_errors=True)
    ex.shutdown(wait=True)
    log("pipeline finished")


if __name__ == "__main__":
    main()
