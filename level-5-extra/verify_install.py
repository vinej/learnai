"""
verify_install.py — sanity check for the Level 5 environment.

Run: python verify_install.py
"""
import importlib
import os
import sys

from dotenv import load_dotenv

load_dotenv()

REQUIRED = [
    "numpy", "pandas", "scipy", "matplotlib", "sklearn", "statsmodels",
    "prophet", "xgboost", "lightgbm", "torch", "darts", "neuralforecast",
    "anthropic", "openai", "yfinance", "fredapi", "mlflow", "fastapi",
    "streamlit",
]

OPTIONAL = ["chronos", "nixtla", "chromadb", "sentence_transformers", "mapie"]


def check(name: str, required: bool = True) -> bool:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "?")
        print(f"  ok   {name:<22} {version}")
        return True
    except ImportError as e:
        marker = "MISS" if required else "skip"
        print(f"  {marker} {name:<22} ({e})")
        return not required


def check_env_var(name: str) -> bool:
    val = os.getenv(name)
    if val:
        print(f"  ok   {name:<22} (set, length={len(val)})")
        return True
    print(f"  MISS {name:<22} (not set in .env)")
    return False


def main() -> int:
    print("=== Required packages ===")
    ok = all(check(p, required=True) for p in REQUIRED)

    print("\n=== Optional packages ===")
    for p in OPTIONAL:
        check(p, required=False)

    print("\n=== API keys ===")
    keys_ok = all(check_env_var(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "FRED_API_KEY"))

    print("\n=== Data smoke test ===")
    try:
        import yfinance as yf
        df = yf.download("SPY", period="5d", progress=False, auto_adjust=True)
        if df.empty:
            print("  MISS yfinance returned empty frame")
            ok = False
        else:
            print(f"  ok   yfinance download    rows={len(df)}")
    except Exception as e:  # noqa: BLE001
        print(f"  MISS yfinance: {e}")
        ok = False

    try:
        from fredapi import Fred
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        s = fred.get_series("CPIAUCSL").tail(3)
        print(f"  ok   FRED CPIAUCSL        latest={s.iloc[-1]:.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"  MISS FRED: {e}")
        keys_ok = False

    print("\n" + ("ALL GOOD" if ok and keys_ok else "Some checks failed — see above."))
    return 0 if ok and keys_ok else 1


if __name__ == "__main__":
    sys.exit(main())
