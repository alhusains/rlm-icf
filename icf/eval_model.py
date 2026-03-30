"""
Custom DeepEval model wrapper for Azure OpenAI.

Allows the evaluation framework to use Azure OpenAI as the LLM judge
without requiring `deepeval set-azure-openai` CLI configuration.
Reads credentials from the same .env file as the rest of the pipeline.
"""

from __future__ import annotations

import os

from deepeval.models import DeepEvalBaseLLM


class AzureOpenAIJudge(DeepEvalBaseLLM):
    """Azure OpenAI judge model for DeepEval GEval metrics.

    Reads connection details from environment variables (loaded via .env):
        AZURE_OPENAI_ENDPOINT
        AZURE_OPENAI_API_KEY
        AZURE_OPENAI_API_VERSION  (default: 2024-12-01-preview)

    Parameters
    ----------
    model_name : str
        Azure deployment name (e.g. "gpt-4o", "gpt-5.1").
    azure_endpoint : str | None
        Override AZURE_OPENAI_ENDPOINT env var.
    api_key : str | None
        Override AZURE_OPENAI_API_KEY env var.
    api_version : str | None
        Override AZURE_OPENAI_API_VERSION env var.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
    ):
        self.model_name = model_name
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._api_version = api_version or os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
        )

        if not self._azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint is required for the evaluation judge. "
                "Set AZURE_OPENAI_ENDPOINT in your .env file or pass azure_endpoint."
            )
        if not self._api_key:
            raise ValueError(
                "Azure OpenAI API key is required for the evaluation judge. "
                "Set AZURE_OPENAI_API_KEY in your .env file or pass api_key."
            )

        # Set before super().__init__ because it calls load_model()
        self._client = None
        self._async_client = None
        super().__init__(model=model_name)

    def load_model(self):
        import openai

        if self._client is None:
            self._client = openai.AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
        return self

    def generate(self, prompt: str, **kwargs) -> str:
        self.load_model()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, **kwargs) -> str:
        import openai

        if self._async_client is None:
            self._async_client = openai.AsyncAzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
        response = await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return self.model_name
