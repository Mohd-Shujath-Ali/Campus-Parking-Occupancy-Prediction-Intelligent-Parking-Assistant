import os
import sys
import io
import json
import base64
import tempfile
from pathlib import Path

# Configure matplotlib headless
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def execute_notebook(nb_path: str, output_path: str = None):
    if output_path is None:
        output_path = nb_path

    print(f"\n=======================================================")
    print(f"Executing: {nb_path}")
    print(f"=======================================================")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Execution namespace
    root_dir = os.path.abspath(".")
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    exec_globals = {
        "__file__": os.path.abspath(nb_path),
        "__name__": "__main__",
    }

    current_outputs = []

    def custom_show(*args, **kwargs):
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            current_outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": b64,
                    "text/plain": [f"<Figure size {fig.get_size_inches()} with {len(fig.axes)} Axes>"]
                },
                "metadata": {}
            })
        plt.close("all")

    def custom_display(*args):
        for arg in args:
            if hasattr(arg, "to_html"):
                current_outputs.append({
                    "output_type": "display_data",
                    "data": {
                        "text/html": [line + "\n" for line in arg.to_html().split("\n")],
                        "text/plain": [str(arg)]
                    },
                    "metadata": {}
                })
            else:
                current_outputs.append({
                    "output_type": "display_data",
                    "data": {
                        "text/plain": [str(arg)]
                    },
                    "metadata": {}
                })

    exec_globals["display"] = custom_display

    cell_counter = 1
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source_code = "".join(cell.get("source", []))
        if not source_code.strip():
            continue

        print(f"Running Code Cell [{cell_counter}]...")
        current_outputs = []

        # Intercept plt.show
        orig_show = plt.show
        plt.show = custom_show

        # Intercept stdout
        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        try:
            exec(source_code, exec_globals)
        except Exception as e:
            sys.stdout = old_stdout
            plt.show = orig_show
            print(f"Error in cell {cell_counter}: {e}")
            import traceback
            tb_str = traceback.format_exc()
            current_outputs.append({
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": tb_str.split("\n")
            })
        else:
            sys.stdout = old_stdout
            plt.show = orig_show

            # Check if any lingering figures were created without calling plt.show()
            if plt.get_fignums():
                custom_show()

            captured = stdout_capture.getvalue()
            if captured:
                current_outputs.insert(0, {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [line + "\n" for line in captured.split("\n") if line]
                })

        cell["execution_count"] = cell_counter
        cell["outputs"] = current_outputs
        cell_counter += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Saved executed notebook with populated outputs -> {output_path}")

if __name__ == "__main__":
    execute_notebook("notebooks/01_exploratory_data_analysis.ipynb")
    execute_notebook("notebooks/02_model_training_and_tuning.ipynb")
