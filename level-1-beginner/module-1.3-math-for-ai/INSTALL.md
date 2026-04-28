# Setup — Module 1.3 (Math for AI)

This module uses **NumPy** for arrays, **Matplotlib** for plots, and **SciPy** for a couple of probability helpers.

## 1. Python ≥ 3.11

See [../module-1.1-python-essentials/INSTALL.md](../module-1.1-python-essentials/INSTALL.md). On Windows, use `py` if `python` isn't on PATH.

## 2. Create a virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

That installs:

| Package | Purpose |
|---------|---------|
| `numpy`      | n-dim arrays, linear algebra, random sampling |
| `matplotlib` | plotting (saved as PNGs in `figures/`) |
| `scipy`      | extra probability distributions and statistics |

## 4. Run the lessons

```bash
python 01_scalars_vectors_matrices.py
python 02_linear_algebra.py
python 03_eigen.py
python 04_calculus_derivatives.py
python 05_gradients_and_chain_rule.py
python 06_probability.py
python 07_statistics.py
python 08_bayes_rule.py
python 09_clt_and_lln.py
```

Each script that produces a figure prints the path of the saved PNG. Open it in your editor — VS Code previews PNGs inline.

## 5. Run the exercises

```bash
python exercises/01_dot_and_matmul_from_scratch.py
python exercises/02_cosine_similarity.py
python exercises/03_gradient_descent.py
python exercises/04_dice_rolls.py
python exercises/05_medical_test_bayes.py
```

## Tips

- Most plots are saved into a `figures/` subfolder. It's auto-created on first run.
- If you want plots to pop up interactively, append `plt.show()` to the bottom of any script (or run it inside a Jupyter notebook).
- This module is math-heavy on the concept side but the code is short. Don't skim the comments — that's the lesson.
