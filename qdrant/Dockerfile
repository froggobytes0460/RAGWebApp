FROM qdrant/qdrant:v1.18@sha256:75eab8c4ba42096724fdcfde8b4de0b5713d529dde32f285a1f86fdcb2c9e50c

COPY certs/server.crt /qdrant/tls/cert.pem
COPY certs/server.key /qdrant/tls/key.pem
RUN chmod 644 /qdrant/tls/cert.pem \
    && chmod 600 /qdrant/tls/key.pem

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=20s \
    CMD printf 'GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n' | \
        openssl s_client -connect 127.0.0.1:6333 -CAfile /qdrant/tls/cert.pem -quiet 2>/dev/null | grep -q '200 OK'
