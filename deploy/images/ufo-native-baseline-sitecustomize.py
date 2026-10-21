"""Only the approved elastic child enables entry observation."""
import os

if os.environ.get("EB_UFO_BASELINE_OBSERVER") == "1":
    from eb.runners.ufo_native_baseline_observer import install
    install()
