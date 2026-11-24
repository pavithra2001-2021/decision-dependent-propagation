#!/bin/bash
set -u
LIST=$1; WID=$2; NW=$3; KEEP=${4:-1}
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
P6="${P6_ROOT:?set P6_ROOT to the WSL path of this folder}"
OUT="${P6_CORR_OUT:-$P6/data/external/derived/corr_nyu2}"     # override only for tests
mkdir -p "$OUT" /root/work/corr_nyu2
cp "$P6/src/p6/dparsf_run.m" "$P6/src/p6/qc_corr.m" /root/tools/
i=0
while IFS='|' read -r s F J T1; do
  T1=${T1%$'\r'}                                   # tolerate CRLF run lists
  [ -z "$s" ] && continue
  for f in "$F" "$J" "$T1"; do [ -f "$f" ] || { echo "$s FAILED missing input $f $(date +%FT%T)" >> "$OUT/p6_staging_log.txt"; continue 2; }; done
  i=$((i + 1))
  [ "$NW" -gt 0 ] || continue                     # runs are claimed dynamically (lock dirs), so any worker may take any run
  [ -f "$OUT/$s/ROISignals_$s.mat" ] && continue
  mkdir "$OUT/.lock_$s" 2>/dev/null || continue
  W=/root/work/corr_nyu2/$s
  rm -rf "$W"; mkdir -p "$W/FunImg/$s" "$W/T1Img/$s"
  cp "$F" "$W/FunImg/$s/"; cp "$J" "$W/FunImg/$s/$(basename "${F%.nii.gz}").json"; cp "$T1" "$W/T1Img/$s/co_T1w.nii.gz"
  python3 - "$W/FunImg/$s/$(basename "${F%.nii.gz}").json" << 'PY'
import json, sys
p = sys.argv[1]; j = json.load(open(p)); st = j.get("SliceTiming")
if st and max(st) > float(j["RepetitionTime"]):
    j["SliceTiming"] = [t / 1000.0 for t in st]; j["P6_SliceTiming_converted_ms_to_s"] = True
    json.dump(j, open(p, "w"), indent=1)
PY
  t0=$(date +%s)
  (cd /root/tools && octave-cli --no-gui --eval "addpath('/root/tools'); dparsf_run('$W', 1, {'$s'}, '/root/tools/DPABI', '/root/tools/spm12'); addpath('/root/tools'); qc_corr('$W', '$s', '/root/tools/spm12')" > "$W/log.txt" 2>&1)
  t1=$(date +%s)
  R="$W/Results/ROISignals_FunImgARCFW/ROISignals_$s.mat"
  if [ -f "$R" ] && grep -q "P6_QC_OK $s" "$W/log.txt"; then
    mkdir -p "$OUT/$s"
    cp "$R" "$OUT/$s/"; cp "$W/RealignParameter/$s/"FD_Jenkinson_*.txt "$OUT/$s/"; cp "$W/RealignParameter/$s/"rp_*.txt "$OUT/$s/"
    cp "$W/log.txt" "$OUT/$s/dparsf_log.txt"; cp "$W/TRInfo.tsv" "$OUT/$s/" 2>/dev/null
    mkdir -p "$OUT/$s/qc"; cp "$W/QC/$s/"wmean*.nii "$W/QC/$s/brainmask_w.nii" "$W/Masks/aal_3mm_resliced.nii" "$OUT/$s/qc/"
    { for st in 1a 1b 1c; do [ -f "$W/P6_stage$st.done" ] && echo "stage$st 1" || echo "stage$st 0"; done
      [ -d "$W/FunImgA/$s" ] && echo "slice_timing_dir 1" || echo "slice_timing_dir 0"; } > "$OUT/$s/qc/stage_flags.txt"
    echo "$s OK $((t1 - t0))s $(date +%FT%T)" >> "$OUT/p6_staging_log.txt"
  else
    echo "$s FAILED $((t1 - t0))s $(date +%FT%T) $(grep -m1 -i 'error' "$W/log.txt")" >> "$OUT/p6_staging_log.txt"
    mkdir -p "$OUT/failed_logs"; cp "$W/log.txt" "$OUT/failed_logs/$s.log"     # diagnosis evidence kept before cleanup
  fi
  rmdir "$OUT/.lock_$s"
  [ "$KEEP" = "0" ] && rm -rf "$W"
done < "$LIST"
echo "WORKER_DONE $WID"
