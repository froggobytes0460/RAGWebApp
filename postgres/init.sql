-- Grant explicit connect privilege (belt-and-suspenders alongside POSTGRES_DB ownership)
GRANT CONNECT ON DATABASE ragdb TO raguser;
