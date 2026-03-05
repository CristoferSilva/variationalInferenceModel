FROM python:3.11

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install jupyter ipykernel

COPY . .

CMD ["sleep", "infinity"]