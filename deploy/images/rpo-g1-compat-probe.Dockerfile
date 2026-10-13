FROM eb/tools:rpo-g1-source-v1
COPY --from=eb/tools:rpo-g1-code-v1 /opt/hhtools/hhtools /opt/hhtools/hhtools
ENV PYTHONPATH=/opt/hhtools:/workspace/eb_runner
