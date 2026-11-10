"""In-process Jupyter Notebook Builder and Executor.
Captures stdout, text outputs, and all matplotlib plots as base64 PNG outputs in .ipynb.
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".mplcache")

import io, sys, base64, traceback
import nbformat as nbf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


class NotebookRunner:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.nb = nbf.v4.new_notebook()
        self.counter = 1
        self.env = {
            "__name__": "__main__",
            "os": os,
            "sys": sys,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
        }

    def md(self, text: str):
        self.nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(self, code_str: str):
        code_clean = code_str.strip()
        cell = nbf.v4.new_code_cell(code_clean)
        cell.execution_count = self.counter
        self.counter += 1

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        figures = []

        def capture_figures():
            for num in plt.get_fignums():
                fig = plt.figure(num)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=100)
                figures.append(base64.b64encode(img_buf.getvalue()).decode("utf-8"))
            plt.close("all")

        # Intercept plt.show so that figures are captured before closing
        old_show = plt.show
        plt.show = capture_figures
        self.env["plt"] = plt
        self.env["plt"].show = capture_figures

        try:
            exec(code_clean, self.env)
            # Capture any remaining figures not explicitly closed by plt.show
            capture_figures()
        except Exception:
            traceback.print_exc(file=buf)
            capture_figures()
        finally:
            plt.show = old_show
            sys.stdout = old_stdout

        out_text = buf.getvalue()
        outputs = []
        if out_text:
            outputs.append(nbf.v4.new_output(output_type="stream", name="stdout", text=out_text))
        for fig_b64 in figures:
            outputs.append(nbf.v4.new_output(output_type="display_data", data={"image/png": fig_b64}))

        cell.outputs = outputs
        self.nb.cells.append(cell)

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            nbf.write(self.nb, f)
        print(f"✓ Saved executed notebook with {len(self.nb.cells)} cells to {self.output_path}")
