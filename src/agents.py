from pydantic_ai import Agent
from pydantic_ai.models import openai
from pydantic_ai.providers import openai

from load_api import Settings

settings = Settings()

class OpenAIAgentInit:
    
    # Agent Creation
    def create_table_agent(sysprompt: str, model_name: str ):
        table_model = openai(
            model_name, provider=openai(api_key=settings.OPENAI_API_KEY)
        )
        
        table_agent = Agent(
            table_model,
            system_prompt=sysprompt,
        )
        
        return table_agent

    def create_img_agent(sysprompt: str, model_name: str):
        image_caption_model = openai(
            model_name, provider=openai(api_key=settings.OPENAI_API_KEY)
        )
        
        image_caption_agent = Agent(
            image_caption_model,
            system_prompt=sysprompt
        )
        
        return image_caption_agent

    def create_chat_agent(sysprompt: str, model_name: str):
        chat_model = openai(
            model_name, provider=openai(api_key=settings.OPENAI_API_KEY)
        )
        
        chat_agent = Agent(
            chat_model,
            system_prompt=sysprompt
        )
        
        return chat_agent