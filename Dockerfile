FROM python:3.10.6
LABEL authors="Peterhzx"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python"]

CMD train.py