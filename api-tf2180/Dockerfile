FROM python:3.12.8

WORKDIR /app

COPY app ./app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY init.sh .
RUN chmod +x init.sh

EXPOSE 8080

ENTRYPOINT ["./init.sh"]