#!/usr/bin/env bash
# Every validation of docs/validation.md from one code state, in one go.
#
#   benchmarks/validate_all.sh OUT_DIR [JOBS] [FIRST_STEP]
#
# FIRST_STEP (sbc, detection, misspec, bias, corr, alfalfa, figures) skips the
# steps before it, e.g. to continue after a change that leaves the SBC valid.
# Runs the SBC (1000), detection (200), misspecification (5 x 100), the
# conditional-bias study (default and flat priors), the correlated-noise
# study and the ALFALFA mini study, then regenerates the figures in docs/img.
# Each summary.json records the git commit. About 6 hours on 4 cores.
set -euo pipefail
out=${1:?output directory}
jobs=${2:-4}
first=${3:-sbc}
mkdir -p "$out"
git -C "$(dirname "$0")/.." rev-parse HEAD >> "$out/commit.txt"
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$out/validate_all.log"; }
steps=(sbc detection misspec bias corr alfalfa figures)
run() { for s in "${steps[@]}"; do [ "$s" = "$first" ] && return 0; [ "$s" = "$1" ] && return 1; done; return 0; }

run sbc       && { log "sbc";       hiprofile validate --stage sbc --n 1000 --jobs "$jobs" --seed 11 --out "$out" > "$out/sbc.log" 2>&1; }
run detection && { log "detection"; hiprofile validate --stage detection --n 200 --jobs "$jobs" --seed 21 --out "$out" > "$out/detection.log" 2>&1; }
run misspec   && { log "misspec";   hiprofile validate --stage misspec --n 100 --jobs "$jobs" --seed 31 --out "$out" > "$out/misspec.log" 2>&1; }
run bias      && { log "bias default"; python benchmarks/bias_study.py --out "$out/bias" --n 20 --jobs "$jobs" --priors default > "$out/bias_default.log" 2>&1
                   log "bias flat";  python benchmarks/bias_study.py --out "$out/bias" --n 20 --jobs "$jobs" --priors flat > "$out/bias_flat.log" 2>&1; }
run corr      && { log "correlated noise"; python benchmarks/correlated_noise.py --out "$out/corr" --n 300 --n-noise 150 --jobs "$jobs" \
                   --figure docs/img/correlated_noise.png > "$out/corr.log" 2>&1; }
run alfalfa   && { log "alfalfa";   python benchmarks/alfalfa_study.py --out "$out/alfalfa" --jobs "$jobs" \
                   --figure docs/img/alfalfa_study.png > "$out/alfalfa.log" 2>&1; }
log "figures";    python docs/make_figures.py validation "$out" > "$out/figures.log" 2>&1
                  python docs/make_figures.py bias "$out/bias" >> "$out/figures.log" 2>&1
                  python docs/make_figures.py gallery >> "$out/figures.log" 2>&1
log "done: $(cat "$out/commit.txt")"
