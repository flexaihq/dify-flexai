ENDPOINT_URL = "https://api.flex.ai/v1"


def apply_endpoint(credentials: dict) -> None:
    """Point the OpenAI-compatible base classes at FlexAI."""
    credentials["endpoint_url"] = ENDPOINT_URL
