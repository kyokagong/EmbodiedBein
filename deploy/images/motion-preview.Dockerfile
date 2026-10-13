FROM eb/tools:latest
COPY backend/eb/motion_preview.py /opt/eb_tools/eb/motion_preview.py
COPY backend/eb/runners/motion_preview.py /opt/eb_tools/eb/runners/motion_preview.py
