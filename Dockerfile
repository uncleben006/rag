FROM python:3.12-slim

RUN pip install poetry==2.0.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./.env ./

COPY ./embedding ./embedding

COPY ./langserveapp/package[s] ./packages

RUN poetry install  --no-interaction --no-ansi --no-root

COPY ./langserveapp/app ./app

RUN poetry install --no-interaction --no-ansi --no-root

EXPOSE 8080

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8080
