# Dockerfile

FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY . /app

EXPOSE 80

CMD ["gunicorn", "-b", "0.0.0.0:80", "app:app"]
