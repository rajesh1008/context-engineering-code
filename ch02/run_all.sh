#!/bin/sh
# Regenerate every printed output of Chapter 2 (output.txt and out/*.txt).
set -e
cd "$(dirname "$0")"
mkdir -p out
python3 lab.py > output.txt
python3 stage_variants.py > out/stage_variants.txt
python3 ablation.py > out/ablation.txt
python3 eval_probes.py > out/eval_probes.txt
python3 retrieval_modes.py > out/retrieval_modes.txt
python3 smoke.py > out/smoke.txt || true
python3 smoke.py --stale > out/smoke_stale.txt || true
python3 trace_det8.py > out/trace_det8.txt
python3 flag.py > out/flag.txt
