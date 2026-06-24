"""Prompt templates for binary polarization classification (task 1)."""

SYSTEM_BINARY = (
    "You are a binary classifier. Output ONLY '1' for polarized or '0' for neutral."
)


def format_binary_messages(sample: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_BINARY},
            {"role": "user", "content": sample["text"]},
            {"role": "assistant", "content": str(sample["polarization"])},
        ]
    }


def format_binary_inference_messages(sample: dict) -> dict:
    return {
        "id": sample["id"],
        "messages": [
            {"role": "system", "content": SYSTEM_BINARY},
            {"role": "user", "content": sample["text"]},
        ],
    }


QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] }}{% endgeneration %}"
    "{% else %}"
    "{{ message['content'] }}"
    "{% endif %}"
    "{{ '\\n' }}"
    "{% endfor %}"
)
