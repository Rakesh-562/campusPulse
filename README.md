# CampusPulse
Five vital signs of a campus, one shared method (decision log + Finding Robustness Matrix).
See PLAN.md for the full plan.

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/download_data.py --pulse all     # or: --pulse oulad
    pytest
    streamlit run app/Home.py

## Rules
1. Anything two pulses need goes in `campuspulse/common/`. Never copy code between pulses.
2. Every pulse module `campuspulse/<pulse>/pipeline.py` exposes `DECISIONS`, `FINDINGS`, `run_pipeline(decisions)`
   (see `academic/pipeline.py`). Convention: a positive finding value means the claim holds.
3. Never commit raw data except UCI dropout, Malaysia survey and NOVANIMAL.
