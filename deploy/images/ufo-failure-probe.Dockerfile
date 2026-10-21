FROM eb/ufo-native:rpo-train-v3
COPY backend/eb/ufo_failure_probe.py /opt/eb-native/eb/ufo_failure_probe.py
COPY backend/eb/runners/ufo_failure_probe.py /opt/eb-native/eb/runners/ufo_failure_probe.py
RUN test -x /usr/bin/timeout && python -c "import sys; assert sys.version_info[:2] == (3,10); import eb.ufo_failure_probe, eb.runners.ufo_failure_probe"
CMD ["python", "-m", "eb.runners.ufo_failure_probe", "--help"]
