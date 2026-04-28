# Module 1.4 — Data Handling with NumPy & Pandas

**Level:** 1 — Beginner
**Estimated time:** 2 weeks

## Goal
Manipulate and explore real datasets confidently. This is the bread-and-butter of every ML/AI workflow.

## Topics
### NumPy
- `ndarray` basics, dtypes, shape, axes
- Indexing, slicing, fancy indexing, boolean masks
- Broadcasting rules
- Vectorized ops vs Python loops (and why it matters)
- Common functions: `mean`, `std`, `argmax`, `where`, `concatenate`, `reshape`

### Pandas
- `Series` and `DataFrame`
- Reading data: `read_csv`, `read_json`, `read_parquet`, `read_sql`
- Selection: `loc`, `iloc`, boolean filtering
- Aggregations: `groupby`, `agg`, `pivot_table`
- Joins: `merge`, `concat`, `join`
- Missing data: `isna`, `fillna`, `dropna`
- Time series basics: `to_datetime`, resampling

### Visualization
- `matplotlib` fundamentals (figure, axes, plot types)
- `seaborn` for statistical plots (`histplot`, `boxplot`, `heatmap`, `pairplot`)

## Exercises
1. Load a CSV with messy data — clean missing values, fix dtypes, normalize columns.
2. Reproduce a specific aggregation (e.g., monthly revenue by category) using `groupby`.
3. Replace a `for` loop with vectorized NumPy and benchmark the speedup.
4. Build a 5-plot exploratory dashboard for a Kaggle dataset.

## Capstone (Level 1)
Pick a public dataset (Titanic, Iris, NYC Taxi, etc.). Produce a notebook with:
- Data loading & cleaning
- 5+ insights backed by visualizations
- A short written summary of findings

## Resources
- NumPy quickstart: https://numpy.org/doc/stable/user/quickstart.html
- Pandas user guide: https://pandas.pydata.org/docs/user_guide/
- Book: *Python for Data Analysis* — Wes McKinney

## Checkpoint
You can load a messy CSV, clean it, group/aggregate it, join it with another table, and produce a clear chart from the result.
