### Run Redis

```sh
docker run -d --name redis -p 6379:6379 redis
```

### Run Celery

```sh
celery -A celery_worker worker --loglevel=info
```

### Run FastAPI

```sh
fastapi dev server.py
```
