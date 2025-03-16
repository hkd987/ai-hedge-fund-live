"""Helper functions for LLM"""

import json
import inspect
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from utils.progress import progress

T = TypeVar('T', bound=BaseModel)

def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory = None
) -> T:
    """
    Makes an LLM call with retry logic, handling both Deepseek and non-Deepseek models.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure
        
    Returns:
        An instance of the specified Pydantic model
    """
    from llm.models import get_model, get_model_info, ModelProvider
    
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)
    
    # For non-JSON support models, we can use structured output
    if not (model_info and not model_info.has_json_mode()):
        # Check if we're using OpenAI and ensure 'json' is in the prompt
        if model_provider == ModelProvider.OPENAI:
            # Get the expected field names from the Pydantic model
            field_names = list(pydantic_model.model_fields.keys())
            json_instruction = f"Return your answer as a JSON object with these exact fields: {', '.join(field_names)}."
            
            # If using a chat message format
            if hasattr(prompt, "messages"):
                # Update the last message to include 'json' if it doesn't already
                last_msg = prompt.messages[-1]
                if hasattr(last_msg, "content") and "json" not in last_msg.content.lower():
                    # Append instruction to return as JSON
                    if isinstance(last_msg.content, str):
                        new_content = f"{last_msg.content}\n\n{json_instruction}"
                        prompt.messages[-1].content = new_content
            # If using a simple string prompt
            elif isinstance(prompt, str) and "json" not in prompt.lower():
                prompt = f"{prompt}\n\n{json_instruction}"
        
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )
    
    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Update status to show we're making the LLM call
            if agent_name:
                progress.update_status(agent_name, None, f"Making LLM call ({attempt + 1}/{max_retries})")
                
            # Call the LLM
            result = llm.invoke(prompt)
            
            # Update status to show completion
            if agent_name:
                progress.update_status(agent_name, None, "LLM call completed")
                
            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_deepseek_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result
                
        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")
            
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)

def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None
    
    return model_class(**default_values)

def extract_json_from_deepseek_response(content: str) -> Optional[dict]:
    """Extracts JSON from Deepseek's markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from Deepseek response: {e}")
    return None
