# Setup — Module 5.2

```bash
pip install statsmodels prophet arch statsforecast
```

`prophet` may need a C++ build toolchain on Windows; if pip fails, install via:

```bash
conda install -c conda-forge prophet
```

`arch` provides GARCH; `statsforecast` provides fast AutoARIMA / ETS.

The shared loaders from Module 5.1 are reused — make sure 5.1 ran at least once so the data cache exists.
