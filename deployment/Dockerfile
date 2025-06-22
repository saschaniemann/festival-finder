FROM python:3.13-slim
EXPOSE 8501

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY assets/favicon.png ./assets/favicon.png
COPY static/og_image.jpg ./scripts/static/og_image.jpg

# needed for open graph tags
RUN python3.13 scripts/set_meta_tags.py
ENV STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

CMD [ "streamlit", "run", "scripts/app.py" ]
