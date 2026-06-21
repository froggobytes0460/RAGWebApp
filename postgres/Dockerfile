FROM postgres:17-alpine@sha256:dc17045ccfd343b49600570ea734b9c4991cf1c3f3302e67df51e3b402dd55c4

COPY certs/server.crt /var/lib/postgresql/server.crt
COPY certs/server.key /var/lib/postgresql/server.key
RUN chown postgres:postgres /var/lib/postgresql/server.crt /var/lib/postgresql/server.key \
    && chmod 600 /var/lib/postgresql/server.key \
    && chmod 644 /var/lib/postgresql/server.crt
