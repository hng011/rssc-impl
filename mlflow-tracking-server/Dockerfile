
FROM ghcr.io/mlflow/mlflow:v2.22.0

WORKDIR /app 

COPY . .

RUN chmod +x init.sh

EXPOSE 8080

ENTRYPOINT ["./init.sh"]