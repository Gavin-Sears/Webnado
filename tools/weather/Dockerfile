FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libeccodes0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY tools/weather/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY tools/weather ./weather
EXPOSE 8080
VOLUME /data
CMD ["python", "weather/service.py", "--host", "0.0.0.0", "--data", "/data", "--origin", "https://YOUR-ACCOUNT.github.io"]
