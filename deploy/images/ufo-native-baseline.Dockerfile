# Build with make images-ufo-native-baseline, which verifies the local e122 base.
# Never retag or rebuild the frozen training image.
FROM eb/ufo-native:rpo-train-v2
COPY backend/eb/ufo_native_baseline.py /opt/eb-native/eb/ufo_native_baseline.py
COPY backend/eb/runners/ufo_native_baseline.py /opt/eb-native/eb/runners/ufo_native_baseline.py
COPY backend/eb/runners/ufo_native_baseline_observer.py /opt/eb-native/eb/runners/ufo_native_baseline_observer.py
COPY deploy/images/ufo-native-baseline-sitecustomize.py /opt/eb-baseline/sitecustomize.py
ENV PYTHONPATH=/opt/eb-baseline:/opt/eb-native
CMD ["python", "-m", "eb.runners.ufo_native_baseline", "--plan", "/baseline-plan.json"]
