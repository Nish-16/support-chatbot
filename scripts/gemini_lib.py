"""
Shared Gemini client + intent-classification call, used by both the
standalone classifier script and the golden-set candidate builder.
Pulled out to one place so there's a single source of truth for the
prompt, retry behavior, and model choice.
"""
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import httpx

load_dotenv()

# gemini-3.6-flash's free tier caps at 20 requests/DAY -- unusable for
# a golden set of 150-250+ examples. gemini-flash-lite-latest has a
# much higher free quota and is cheap/fast, which is what actually
# matters for a classification task like this (we don't need the
# biggest model to pick 1 of 8 categories).
MODEL = "gemini-flash-lite-latest"

with open("intents.json") as f:
    INTENTS = json.load(f)

INTENT_NAMES = [i["name"] for i in INTENTS]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENT_NAMES},
        "confidence": {
            "type": "number",
            "description": "0.0 to 1.0, how confident you are this is the single best-fitting intent",
        },
        "reason": {
            "type": "string",
            "description": "one sentence explaining why this intent fits, referencing the message",
        },
    },
    "required": ["intent", "confidence", "reason"],
}

INTENT_LIST_TEXT = "\n".join(f"- {i['name']}: {i['description']}" for i in INTENTS)

PROMPT_TEMPLATE = """You are classifying a customer support tweet sent to Dropbox's support account.

Pick exactly ONE intent from this list that best fits the customer's PRIMARY need \
(the thing they most need resolved -- tweets are often messy and touch more than one topic):

{intent_list}

Customer tweet:
\"\"\"{text}\"\"\"

Return your classification."""

RETRYABLE_NETWORK_ERRORS = (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, APIError):
        return exc.code in (429, 500, 502, 503, 504)
    return isinstance(exc, RETRYABLE_NETWORK_ERRORS)


def make_client() -> genai.Client:
    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=30_000),  # ms
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(_is_retryable),
    reraise=True,  # raise the ORIGINAL exception on final failure, not
                   # tenacity's own RetryError wrapper -- otherwise
                   # `except SKIPPABLE_ERRORS` at call sites never
                   # matches and the whole batch crashes anyway
)
def classify_message(client: genai.Client, text: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(intent_list=INTENT_LIST_TEXT, text=text)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )
    return json.loads(response.text)


# both APIError and the network errors above are worth catching the
# same way at the call site: retries already happened inside
# classify_message, so if we still got one of these, it's time to
# skip this example rather than crash the whole batch
SKIPPABLE_ERRORS = (APIError, *RETRYABLE_NETWORK_ERRORS)
