"""
05 — Logging basics

`print()` is fine for quick scripts, but in real code you want:
  • severity levels (debug / info / warning / error / critical)
  • timestamps and module names automatically attached
  • on/off control without editing every print() call
  • output going to files, services, or stdout — your choice

That's `logging`.

Run:
    python 05_logging_basics.py
"""

import logging

# --- Configure once, at the start of your program ---
# In libraries you should NOT call basicConfig — leave configuration to the app.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Convention: get a logger named after the current module.
log = logging.getLogger(__name__)


def divide(a: float, b: float) -> float:
    log.debug("dividing %s by %s", a, b)        # only printed when DEBUG enabled
    if b == 0:
        log.error("attempted to divide by zero (a=%s)", a)
        raise ValueError("b must not be zero")
    result = a / b
    log.info("result is %s", result)
    return result


def main() -> None:
    log.info("starting")
    log.warning("this is a warning — something looks off but we're continuing")

    try:
        divide(10, 2)
        divide(10, 0)
    except ValueError:
        log.exception("divide failed")    # logs the traceback automatically

    log.info("done")


if __name__ == "__main__":
    main()


# ── Tips ─────────────────────────────────────────────────────────────────────
#
# • Use %-style placeholders (`log.info("x=%s", x)`) rather than f-strings.
#   The string is only formatted if the level is actually enabled — saves
#   work in hot paths.
#
# • `log.exception(msg)` is shorthand for `log.error(msg, exc_info=True)`
#   inside an `except` block. It includes the traceback.
#
# • Levels (low → high): DEBUG, INFO, WARNING, ERROR, CRITICAL.
#   Set `level=logging.INFO` in basicConfig to silence DEBUG.
#
# • For structured logging (JSON for log aggregators), look at
#   `structlog` or `python-json-logger`.
# ──────────────────────────────────────────────────────────────────────────────
