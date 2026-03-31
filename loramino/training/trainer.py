from loramino.training.client import ServiceClient, TrainingClient
from loramino.training.runtime import setup_model as build_model


def setup_model(config_options: dict):
    return build_model(config_options)


def create_training_client(config_options: dict) -> TrainingClient:
    service_client = ServiceClient()
    return service_client.create_lora_training_client(config_options)


def train(config_options: dict) -> dict:
    training_client = create_training_client(config_options)
    return training_client.fit()
