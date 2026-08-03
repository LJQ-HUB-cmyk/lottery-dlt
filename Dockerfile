FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    TZ=UTC

WORKDIR /app

# 先装依赖，利用 Docker 层缓存——改代码不必重装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/

# 数据与缓存挂载出去，容器重建不丢预测记录
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health',timeout=8)"

# 预测计算是 CPU 密集的，用 sync worker；缓存命中后单请求 <5ms
CMD ["sh", "-c", "gunicorn -w ${WEB_CONCURRENCY:-3} -b 0.0.0.0:${PORT} \
     --timeout 120 --graceful-timeout 30 --access-logfile - --error-logfile - \
     src.wsgi:app"]
