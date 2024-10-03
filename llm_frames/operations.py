from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Union, Dict, Type


class SemanticOperation(BaseModel):
    """
    SemanticOperation is a template for defining operations to be applied to a DataFrame.
    """

    # excludes protected namespace: `model_`
    model_config = ConfigDict(protected_namespaces=())

    input_column: str
    output_column: str
    prompt_message: str
    system_message: Optional[str] = None
    model_name: Optional[str] = None
    model_params: Optional[dict] = None
    response_format: Optional[Type[BaseModel]] = None

    def construct_messages(self, context: str) -> dict:
        """Constructs a list of messages to be sent to the OpenAI API."""

        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})

        full_prompt = f"{self.prompt_message} Context: {context}"
        messages.append({"role": "user", "content": full_prompt})
        return messages
    
    def construct_batch_request(self, context: str, custom_id: str) -> dict:
        """Constructs a single request to be included in batch (OpenAI Async API)."""
        
        request_included_in_batch = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model_name,
                "messages": self.construct_messages(context),
            }
        }
        if self.model_params:
            for param_name, param_value in self.model_params.items():
                request_included_in_batch["body"][param_name] = param_value 

        if self.response_format:
            request_included_in_batch["body"]["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": self.response_format.model_json_schema(),
                    "name": self.response_format.__name__,
                    "strict": True,
                }
            }
        return request_included_in_batch


class TranslationOperation(SemanticOperation):
    """
    Example of a SemanticOperation that translates text from one language to another.
    """
    input_column: str = Field("review", description="The name of the column containing the text to be translated.")
    target_language: str = Field("pl", description="The language to translate the text into.")

    def __init__(self, input_column: str, target_language: str, **params):

        super().__init__(
            input_column=input_column,
            output_column=params.get("output_column") or "review_pl",
            prompt_message=f"Translate the following text to {target_language}.",
            model_name=params.get("input_column") or "gpt-4o-mini",
            response_format=params.get("input_column"),
        )



# {
#     "model": "gpt-4o-2024-08-06",
#     "messages": [
#       {
#         "role": "system",
#         "content": "You are a helpful math tutor. Guide the user through the solution step by step."
#       },
#       {
#         "role": "user",
#         "content": "how can I solve 8x + 7 = -23"
#       }
#     ],
#     "response_format": {
#       "type": "json_schema",
#       "json_schema": {
#         "name": "math_reasoning",
#         "schema": {
#           "type": "object",
#           "properties": {
#             "steps": {
#               "type": "array",
#               "items": {
#                 "type": "object",
#                 "properties": {
#                   "explanation": { "type": "string" },
#                   "output": { "type": "string" }
#                 },
#                 "required": ["explanation", "output"],
#                 "additionalProperties": false
#               }
#             },
#             "final_answer": { "type": "string" }
#           },
#           "required": ["steps", "final_answer"],
#           "additionalProperties": false
#         },
#         "strict": true
#       }
#     }
#   }