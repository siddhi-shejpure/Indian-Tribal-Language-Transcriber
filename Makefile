.PHONY: build up down logs shell clean help

DOCKER_COMPOSE = docker-compose
IMAGE_NAME = indian-language-transcriber

help:
	@echo "Available commands:"
	@echo "  make build      - Build Docker image"
	@echo "  make up         - Start the services with docker-compose"
	@echo "  make down       - Stop the services"
	@echo "  make logs       - Show logs"
	@echo "  make shell      - Open a shell in the container"
	@echo "  make clean      - Remove all containers, volumes and images related to this project"
	@echo "  make help       - Show this help"

build:
	$(DOCKER_COMPOSE) build

up:
	$(DOCKER_COMPOSE) up -d
	@echo "Application started at http://localhost:7860"

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs -f

shell:
	$(DOCKER_COMPOSE) exec transcriber /bin/bash

clean: down
	docker rmi $(IMAGE_NAME)
	docker volume prune -f
	docker system prune -f 