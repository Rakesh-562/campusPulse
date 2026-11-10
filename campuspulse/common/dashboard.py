"""One generic renderer so every pulse page looks the same and has an interactive decision toggle."""
import importlib
import streamlit as st
from campuspulse.common.decisions import robustness_matrix, heatmap

PULSES = ["academic", "engagement", "wellbeing", "food", "energy"]


def load_pulse(name):
    try:
        return importlib.import_module(f"campuspulse.{name}.pipeline")
    except Exception:
        return None


def is_ready(mod) -> bool:
    return bool(mod and getattr(mod, "DECISIONS", None))


def render_pulse_page(name: str, title: str, question: str):
    st.title(title); st.caption(question)
    mod = load_pulse(name)
    if not is_ready(mod):
        st.info("Not implemented yet. Copy campuspulse/academic/pipeline.py as a template."); return
    cfg = {k: st.sidebar.selectbox(k, v) for k, v in mod.DECISIONS.items()}
    try:
        res = mod.run_pipeline(cfg)
    except FileNotFoundError as e:
        st.warning(f"Data missing: {e}. Run: python scripts/download_data.py --pulse {name}"); return
    for fid, claim in mod.FINDINGS.items():
        st.metric(f"{fid}: {claim}", f"{res[fid]:.3f}")
    with st.expander("Robustness Matrix", expanded=True):
        vals, surv = robustness_matrix(mod.run_pipeline, mod.DECISIONS, mod.FINDINGS)
        st.pyplot(heatmap(vals, surv, f"{name}: does the finding survive?"))
