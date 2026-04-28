# AI Engineering Course with Python — Beginner to Expert

A structured 4-level curriculum. Each level builds on the previous.

> Each module title below links to its folder. Folders contain a `README.md` (lesson plan), `INSTALL.md` (setup), numbered concept files, and an `exercises/` subfolder.

---

## LEVEL 1 — Beginner (Foundations) — ~6-8 weeks

**Goal:** Be comfortable with Python and the math/data fundamentals AI relies on.

### [Module 1.1 — Python Essentials](level-1-beginner/module-1.1-python-essentials/)
- Variables, types, operators, control flow (`if`, `for`, `while`)
- Functions, scope, lambdas, comprehensions
- Data structures: `list`, `dict`, `set`, `tuple`
- File I/O, exceptions, modules & packages
- Virtual environments (`venv`, `pip`), `pyproject.toml`

### [Module 1.2 — Tooling](level-1-beginner/module-1.2-tooling/)
- VS Code / Jupyter notebooks
- Git & GitHub basics
- Debugging with `pdb` / IDE debugger
- Linting & formatting (`ruff`, `black`)

### [Module 1.3 — Math for AI (applied, not theoretical)](level-1-beginner/module-1.3-math-for-ai/)
- Linear algebra: vectors, matrices, dot product
- Calculus intuition: derivatives, gradients
- Probability: distributions, Bayes' rule
- Statistics: mean, variance, correlation

### [Module 1.4 — Data Handling with NumPy & Pandas](level-1-beginner/module-1.4-numpy-pandas/)
- `numpy` arrays, broadcasting, vectorization
- `pandas` DataFrames: filtering, grouping, joining
- Reading CSV/JSON/Parquet
- Basic plotting with `matplotlib` / `seaborn`

**Capstone:** Exploratory data analysis on a public dataset (e.g., Titanic, Iris).

---

## LEVEL 2 — Intermediate (Classical ML) — ~8-10 weeks

**Goal:** Train, evaluate, and reason about traditional ML models.

### [Module 2.1 — ML Concepts](level-2-intermediate/module-2.1-ml-concepts/)
- Supervised vs unsupervised vs reinforcement learning
- Train/validation/test splits, cross-validation
- Bias-variance tradeoff, overfitting/underfitting
- Metrics: accuracy, precision/recall, F1, ROC-AUC, RMSE, MAE

### [Module 2.2 — Scikit-learn](level-2-intermediate/module-2.2-scikit-learn/)
- Linear & logistic regression
- Decision trees, random forests, gradient boosting (XGBoost, LightGBM)
- k-NN, SVM, k-means, PCA
- Pipelines, `ColumnTransformer`, hyperparameter search (`GridSearchCV`, `Optuna`)

### [Module 2.3 — Feature Engineering](level-2-intermediate/module-2.3-feature-engineering/)
- Encoding categoricals, scaling, imputation
- Feature selection, target encoding
- Handling imbalanced data (SMOTE, class weights)
- Time-series features

### [Module 2.4 — Experiment Tracking](level-2-intermediate/module-2.4-experiment-tracking/)
- Logging runs with MLflow or Weights & Biases
- Versioning data & models
- Reproducibility (seeds, env locks)

**Capstone:** End-to-end Kaggle-style project with a trained, evaluated, and tracked model.

---

## LEVEL 3 — Advanced (Deep Learning & Specializations) — ~10-12 weeks

**Goal:** Build neural networks for vision, text, and structured data.

### [Module 3.1 — Deep Learning Foundations](level-3-advanced/module-3.1-deep-learning-foundations/)
- Perceptrons, MLPs, activation functions
- Backpropagation & gradient descent variants (SGD, Adam)
- Loss functions, regularization (dropout, weight decay, batch norm)
- PyTorch fundamentals: tensors, autograd, `nn.Module`, training loop

### [Module 3.2 — Computer Vision](level-3-advanced/module-3.2-computer-vision/)
- CNNs: convolutions, pooling, architectures (ResNet, EfficientNet)
- Transfer learning with `torchvision`
- Data augmentation (`albumentations`)
- Object detection & segmentation (YOLO, Mask R-CNN basics)

### [Module 3.3 — NLP & Sequence Models](level-3-advanced/module-3.3-nlp-sequence-models/)
- Tokenization, embeddings (word2vec, GloVe)
- RNN, LSTM, GRU
- Transformers: attention, encoder/decoder
- Hugging Face `transformers`: fine-tuning BERT for classification

### [Module 3.4 — Other Architectures](level-3-advanced/module-3.4-other-architectures/)
- Autoencoders, VAEs
- GANs (intuition + simple implementation)
- Recommenders (matrix factorization, two-tower models)

### [Module 3.5 — Training at Scale](level-3-advanced/module-3.5-training-at-scale/)
- GPU usage, mixed precision (`torch.amp`)
- Distributed training (DDP basics)
- Profiling & optimization

**Capstone:** Fine-tune a transformer or train a vision model on a domain-specific dataset and ship a Streamlit/Gradio demo.

---

## LEVEL 4 — Expert (AI Engineering & Production) — ~10-14 weeks

**Goal:** Build, deploy, and operate production-grade AI systems — including LLM apps and agents.

### [Module 4.1 — LLMs & Prompt Engineering](level-4-expert/module-4.1-llms-prompt-engineering/)
- LLM architectures (decoder-only transformers)
- Prompting patterns: few-shot, chain-of-thought, structured outputs
- Working with the Anthropic / OpenAI SDKs in Python
- Prompt caching, streaming, token & cost accounting

### [Module 4.2 — Retrieval-Augmented Generation (RAG)](level-4-expert/module-4.2-rag/)
- Embeddings & vector databases (FAISS, pgvector, Pinecone, Chroma)
- Chunking strategies, hybrid search, reranking
- Frameworks: LangChain, LlamaIndex (and when to skip them)
- Evaluating RAG: faithfulness, answer relevance, context precision

### [Module 4.3 — Fine-tuning & Adapters](level-4-expert/module-4.3-fine-tuning/)
- Full fine-tuning vs LoRA / QLoRA / PEFT
- Instruction tuning, RLHF / DPO conceptually
- Quantization (GPTQ, AWQ, bitsandbytes)
- Evaluation: benchmarks, LLM-as-judge, human eval

### [Module 4.4 — Agents & Tool Use](level-4-expert/module-4.4-agents-tool-use/)
- Tool/function calling
- Agent loops, planners, ReAct pattern
- Multi-agent orchestration
- Safety: guardrails, sandboxing, prompt injection defenses

### [Module 4.5 — MLOps & Deployment](level-4-expert/module-4.5-mlops-deployment/)
- Serving: FastAPI, BentoML, Triton, vLLM, TGI
- Containerization (Docker), Kubernetes basics
- CI/CD for ML (GitHub Actions, model registries)
- Feature stores (Feast), pipelines (Airflow, Prefect, Dagster)
- Monitoring: drift, latency, cost, hallucination rate

### [Module 4.6 — Responsible & Scalable AI](level-4-expert/module-4.6-responsible-scalable-ai/)
- Bias, fairness, explainability (SHAP, LIME)
- Privacy (differential privacy, PII redaction)
- Cost/latency optimization (batching, caching, model cascades)
- Security (OWASP LLM Top 10)

**Capstone:** Ship a production LLM application — RAG + agents + monitoring + evals — deployed behind an API with CI/CD.

---

## Suggested Stack Across All Levels

| Area | Tools |
|------|-------|
| Core | Python, NumPy, Pandas |
| Classical ML | scikit-learn, XGBoost, LightGBM |
| Deep Learning | PyTorch, Hugging Face |
| LLMs | Anthropic SDK, LangChain/LlamaIndex, vLLM |
| MLOps | MLflow, W&B, Docker, FastAPI, GitHub Actions |
| Data | DuckDB, Postgres+pgvector, Parquet |
