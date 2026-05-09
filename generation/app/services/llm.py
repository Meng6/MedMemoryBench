"""LLM service using LiteLLM."""
import asyncio
import json
import logging
import os
import time
from typing import Optional
import litellm
from litellm import acompletion

from ..config import get_settings

settings = get_settings()

TEMPERATURE = 1.0
DEFAULT_TIMEOUT = settings.llm_timeout  # Read default timeout from config
MAX_RETRY_ATTEMPTS = 10  # Maximum retry attempts (timeout retries)
RETRY_BASE_DELAY = 5  # Base retry delay (seconds)

# Configure logging
logger = logging.getLogger(__name__)

# Configure LiteLLM
litellm.set_verbose = False

# Set API base if configured
if settings.openai_api_base:
    os.environ["OPENAI_API_BASE"] = settings.openai_api_base


class LLMService:
    """Service for LLM interactions using LiteLLM."""

    def __init__(self, model: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        """Initialize LLM service.

        Args:
            model: LLM model to use. Defaults to settings.llm_model.
            timeout: Request timeout in seconds.
        """
        self.model = model or settings.llm_model
        self.timeout = timeout

    async def complete(
        self,
        messages: list[dict],
        temperature: float = TEMPERATURE,
        max_tokens: int = 2000,
        json_mode: bool = False,
        caller: str = "unknown",
        max_retries: int = 3,
        timeout: Optional[int] = None,
        retry_on_timeout: bool = True,
    ) -> str:
        """Send a completion request to the LLM.

        Args:
            messages: List of message dicts with role and content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: If True, request JSON output.
            caller: Identifier for the calling function (for audit logging).
            max_retries: Maximum number of retries on empty response.
            timeout: Request timeout in seconds (overrides instance default).
            retry_on_timeout: If True, retry on timeout until success (up to MAX_RETRY_ATTEMPTS).

        Returns:
            The assistant's response content.

        Raises:
            ValueError: If response is empty after all retries.
            asyncio.TimeoutError: If request times out after all retry attempts.
        """
        request_timeout = timeout or self.timeout

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": request_timeout,  # LiteLLM supports timeout parameter
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error = None
        timeout_retry_count = 0  # Timeout retry counter (independent of empty-response retry)

        for attempt in range(max_retries):
            while True:  # Timeout retry loop
                # Log request
                start_time = time.time()
                retry_info = ""
                if attempt > 0:
                    retry_info = f" empty_retry={attempt + 1}/{max_retries}"
                if timeout_retry_count > 0:
                    retry_info += f" timeout_retry={timeout_retry_count}/{MAX_RETRY_ATTEMPTS}"

                logger.info(
                    f"[LLM Request] model={self.model} caller={caller} "
                    f"messages={len(messages)} max_tokens={max_tokens} json_mode={json_mode} timeout={request_timeout}s"
                    + retry_info
                )
                # Log full input messages
                logger.debug(f"[LLM Request] caller={caller} input_messages={json.dumps(messages, ensure_ascii=False)}")

                try:
                    # Use asyncio.wait_for to add timeout protection
                    response = await asyncio.wait_for(
                        acompletion(**kwargs),
                        timeout=request_timeout + 10  # Leave a small buffer for LiteLLM internal timeout
                    )
                    elapsed = time.time() - start_time

                    # Extract usage info
                    usage = response.usage
                    content = response.choices[0].message.content

                    logger.info(
                        f"[LLM Response] model={self.model} caller={caller} "
                        f"elapsed={elapsed:.2f}s prompt_tokens={usage.prompt_tokens} "
                        f"completion_tokens={usage.completion_tokens} total_tokens={usage.total_tokens}"
                    )
                    # Log full output content
                    logger.debug(f"[LLM Response] caller={caller} output_content={content}")

                    # Check for empty response
                    if content is None or (isinstance(content, str) and content.strip() == ""):
                        logger.warning(
                            f"[LLM Empty Response] model={self.model} caller={caller} "
                            f"attempt={attempt + 1}/{max_retries} - Response content is empty, retrying..."
                        )
                        last_error = ValueError(f"LLM returned empty response after {attempt + 1} attempts")
                        # Add a small delay before retry
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                        break  # Break out of timeout retry loop, enter next empty-response retry

                    return content

                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    timeout_retry_count += 1

                    if retry_on_timeout and timeout_retry_count < MAX_RETRY_ATTEMPTS:
                        # Compute backoff delay: exponential backoff, max 60s
                        delay = min(RETRY_BASE_DELAY * (2 ** (timeout_retry_count - 1)), 60)
                        logger.warning(
                            f"[LLM Timeout] model={self.model} caller={caller} "
                            f"elapsed={elapsed:.2f}s timeout={request_timeout}s - "
                            f"Request timed out, retrying in {delay}s... "
                            f"(attempt {timeout_retry_count}/{MAX_RETRY_ATTEMPTS})"
                        )
                        await asyncio.sleep(delay)
                        continue  # Continue timeout retry
                    else:
                        logger.error(
                            f"[LLM Timeout Failed] model={self.model} caller={caller} "
                            f"elapsed={elapsed:.2f}s - All {MAX_RETRY_ATTEMPTS} timeout retries exhausted"
                        )
                        raise asyncio.TimeoutError(
                            f"LLM request timed out after {timeout_retry_count} retries"
                        )

                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.error(
                        f"[LLM Error] model={self.model} caller={caller} "
                        f"elapsed={elapsed:.2f}s error={type(e).__name__}: {str(e)}"
                    )
                    last_error = e
                    # Retry on exception as well
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))
                    break  # Break out of timeout retry loop

        # All retries exhausted with empty responses
        logger.error(
            f"[LLM Empty Response Failed] model={self.model} caller={caller} "
            f"All {max_retries} retries exhausted - Response consistently empty"
        )
        raise last_error or ValueError(f"LLM returned empty response after {max_retries} retries")

    async def complete_json(
        self,
        messages: list[dict],
        temperature: float = TEMPERATURE,
        max_tokens: int = 2000,
        caller: str = "unknown",
        max_retries: int = 2,
        timeout: Optional[int] = None,
    ) -> dict:
        """Send a completion request and parse JSON response.

        Args:
            messages: List of message dicts with role and content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            caller: Identifier for the calling function (for audit logging).
            max_retries: Maximum number of retries on JSON parse failure.
            timeout: Request timeout in seconds (overrides instance default).

        Returns:
            Parsed JSON response as dict.
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                content = await self.complete(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=True,
                    caller=caller,
                    timeout=timeout,
                )

                # Clean JSON content (strip possible markdown code block markers)
                cleaned_content = self._clean_json_content(content)

                # Try to parse JSON
                parsed_json = json.loads(cleaned_content)

                # Validate parsed result is a valid dict
                if not isinstance(parsed_json, dict):
                    raise ValueError(f"Parsed JSON is not a dict, got {type(parsed_json)}")

                logger.info(f"[LLM] JSON parse success - caller={caller}, attempt={attempt + 1}")
                return parsed_json

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                logger.warning(
                    f"[LLM] JSON parse failed (attempt {attempt + 1}/{max_retries}）- caller={caller}\n"
                    f"Original content (first 500 chars): {content[:500] if 'content' in locals() else 'N/A'}\n"
                    f"Error: {str(e)}"
                )

                # If not the last attempt, add stronger constraint prompt
                if attempt < max_retries - 1:
                    messages_with_emphasis = messages.copy()
                    if messages_with_emphasis and messages_with_emphasis[-1]["role"] == "user":
                        messages_with_emphasis[-1]["content"] += (
                            "\n\n⚠️ Important reminder: you must only return pure JSON, "
                            "Do not include any extra text, explanation, markdown markers or comments."
                            "Start directly with { or [ and end with } or ]."
                        )
                    messages = messages_with_emphasis

                    # Slightly lower temperature for more stable output
                    temperature = max(0.1, temperature - 0.2)

        # All retries failed
        logger.error(
            f"[LLM] JSON parse failed (all retries exhausted)- caller={caller}\n"
            f"Last error: {str(last_error)}"
        )
        raise last_error

    def _clean_json_content(self, content: str) -> str:
        """Clean JSON content, removing possible markdown markers and extra whitespace.

        Args:
            content: Original content

        Returns:
            Cleaned JSON string
        """
        if not content:
            return "{}"

        content = content.strip()

        # Remove leading markdown code block markers
        if content.startswith("```json"):
            content = content[7:].strip()
        elif content.startswith("```"):
            content = content[3:].strip()

        # Remove trailing markdown code block markers
        if content.endswith("```"):
            content = content[:-3].strip()

        # Strip leading text before JSON; find first { or [ as JSON start
        json_start = -1
        for i, char in enumerate(content):
            if char in ['{', '[']:
                json_start = i
                break

        if json_start > 0:
            content = content[json_start:].strip()
        elif json_start == -1:
            # No JSON start symbol found, try returning empty object
            logger.warning(f"[LLM] JSON start symbol not found, first 100 chars of content: {content[:100]}")
            return "{}"

        # Find matching closing } or ] as JSON end
        json_end = -1
        bracket_stack = []
        for i, char in enumerate(content):
            if char in ['{', '[']:
                bracket_stack.append(char)
            elif char == '}':
                if bracket_stack and bracket_stack[-1] == '{':
                    bracket_stack.pop()
                    if not bracket_stack:  # All brackets matched
                        json_end = i
            elif char == ']':
                if bracket_stack and bracket_stack[-1] == '[':
                    bracket_stack.pop()
                    if not bracket_stack:  # All brackets matched
                        json_end = i

        # If a complete JSON structure was found, truncate to that position
        if json_end >= 0:
            content = content[:json_end + 1].strip()

        return content


# Global LLM service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
