FROM qdrant/qdrant:v1.18@sha256:75eab8c4ba42096724fdcfde8b4de0b5713d529dde32f285a1f86fdcb2c9e50c

COPY certs/server.crt /qdrant/tls/cert.pem
COPY certs/server.key /qdrant/tls/key.pem
RUN chown qdrant:qdrant /qdrant/tls/cert.pem /qdrant/tls/key.pem \
    && chmod 644 /qdrant/tls/cert.pem \
    && chmod 600 /qdrant/tls/key.pem
