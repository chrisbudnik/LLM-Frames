from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
import pandas as pd
from typing import List, Dict
from .operations import SemanticOperation
import tiktoken


class LLMFrame:
    def __init__(self, df: pd.DataFrame, client: OpenAI):
        self.df = df
        self.client = client

    def extend(self, operations: List[SemanticOperation]):
        """Creates new columns in the DataFrame by applying the provided operations."""

        for operation in operations:
            self.df[operation.output_column] = self.df.apply(
                lambda row: self._apply_operation(operation, row), axis=1
            )

    def _apply_operation(self, operation: SemanticOperation, row: pd.Series) -> str:
        """Applies the operation to a single row of the DataFrame."""

        context = row[operation.input_column]
        messages = operation.construct_messages(context)
        
        response: ChatCompletion = self.client.chat.completions.create(
            model=operation.model_name or "gpt-4o-mini",
            messages=messages,
            **operation.model_params or {}
        )
        return response.choices[0].message.content
    
    def create_batch_file(
            self,
            batch_file_name: str,
            operation: SemanticOperation, 
            id_column: str
        ) -> None:
        """Creates a batch jsonl file for the OpenAI Async API."""

        assert len(self.df) < 50_000, "Maximum number of requests per batch is 50,000."

        with open(batch_file_name, "w") as f:
            for idx, row in self.df.iterrows():
                batch = operation.construct_batch_request(
                    context=row[operation.input_column], 
                    custom_id=row[id_column]
                )
                f.write(str(batch) + "\n")

        print(f"Batch file created: {batch_file_name}")

    def estimate_input_token_usage(self, operation: SemanticOperation) -> Dict:
        """Estimates the token usage for an operation on the DataFrame."""

        token_counts: pd.Series = self.df.apply(
            lambda row: self._count_token_usage(row[operation.input_column], operation.model_name),
            axis=1
        )
        token_usage = {
            "total_tokens": token_counts.sum(),
            "average_tokens": token_counts.mean(),
            "max_tokens": token_counts.max(),
            "min_tokens": token_counts.min()
        }
        return token_usage

    def _count_token_usage(self, text: str, model_name: str) -> int:
        """Returns the number of tokens in a text string."""
        
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(text))
        return num_tokens



