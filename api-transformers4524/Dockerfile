FROM python:3.12.8

WORKDIR /app

COPY app ./app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY init.sh .
RUN chmod +x init.sh

EXPOSE 8080

ENTRYPOINT ["./init.sh"]