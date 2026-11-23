ARG FOAMNEST_RUNTIME_IMAGE=python:3.12-slim
FROM ${FOAMNEST_RUNTIME_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 FOAMNEST_DATA=/var/lib/foamnest
RUN if [ -d /app ]; then rm -rf /app; fi
WORKDIR /opt/foamnest
COPY requirements.txt scripts/check_dependencies.py ./
RUN (python check_dependencies.py || python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt) \
    && python check_dependencies.py && python -m pip check
COPY foamnest ./foamnest
COPY image_lab ./image_lab
COPY static ./static
COPY templates ./templates
RUN useradd -U -u 11535 foamnest && install -d -o foamnest -g foamnest /var/lib/foamnest
USER foamnest
EXPOSE 8000
HEALTHCHECK NONE
CMD ["python", "-m", "foamnest"]
