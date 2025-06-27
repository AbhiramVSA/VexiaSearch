from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from load_api import Settings

settings = Settings()

class OpenAIAgentInit:
    
    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_table_agent(self, sysprompt: str, model_name: str) -> Agent:
        """Creates an agent for summarizing tables."""
        table_model = OpenAIModel(
            model_name, provider=OpenAIProvider(api_key=self.api_key)
        )
        return Agent(table_model, system_prompt=sysprompt)

    def create_img_agent(self, sysprompt: str, model_name: str) -> Agent:
        """Creates an agent for captioning images."""
        image_caption_model = OpenAIModel(
            model_name, provider=OpenAIProvider(api_key=self.api_key)
        )
        return Agent(image_caption_model, system_prompt=sysprompt)

    def create_chat_agent(self, sysprompt: str, model_name: str) -> Agent:
        """Creates a generic chat agent."""
        chat_model = OpenAIModel(
            model_name, provider=OpenAIProvider(api_key=self.api_key)
        )
        return Agent(chat_model, system_prompt=sysprompt)