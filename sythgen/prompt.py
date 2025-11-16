from topics import topics

USER_PROMPT_TEMPLATE = """
INSTRUCTIONS – FOLLOW STRICTLY STEP BY STEP AND ENSURE HIGH PRECISION:

1. Carefully analyze the topic "{topic}" from the perspective of a user, focusing on the underlying goal, question, or problem the user wants to solve.
2. Identify EXACTLY 10 distinct user intents related to "{topic}". Each intent must represent a **unique, actionable, or knowledge-seeking goal** a user may have.
3. For EACH user intent, generate:
   a) EXACTLY ONE **SPECIFIC QUERY** that is clear, precise, and likely to return highly relevant results. Assign binary_label = 1.
   b) EXACTLY ONE **AMBIGUOUS QUERY** that is vague, imprecise, or open to multiple interpretations. Assign binary_label = 0.
4. Make sure the specific query **directly answers or addresses the intent**, while the ambiguous query could lead to varied or unclear results.
5. OUTPUT FORMAT – USE TAB CHARACTERS (\t) ONLY:
   topic[TAB]query[TAB]binary_label[TAB]user_information_need
6. OUTPUT REQUIREMENTS:
   - Produce EXACTLY 20 ROWS PER TOPIC (2 rows per user intent – one specific, one ambiguous).
   - DO NOT include headers, notes, explanations, or any extra text.
   - EACH ROW MUST FOLLOW THE TAB-DELIMITED FORMAT WITHOUT DEVIATION.
7. QUALITY CHECKS (MANDATORY):
   - Specific queries MUST have binary_label = 1.
   - Ambiguous queries MUST have binary_label = 0.
   - Each user intent MUST appear twice: once with a specific query, once with an ambiguous query.
   - Queries must be grammatically correct and contextually relevant.
8. FINAL OUTPUT:
   - EXACTLY 20 rows for the topic.
   - Each row fully adheres to the tab-delimited format: topic[TAB]query[TAB]binary_label[TAB]user_information_need.
"""


def get_user_prompt(topic: str) -> str:
    return USER_PROMPT_TEMPLATE.format(topic=topic)


SYSTEM_PROMPT = "You are an extremely precise and obedient assistant. Follow **exactly** what the user instructs, step by step, without adding, omitting, or altering instructions. Do not provide explanations, assumptions, or extra commentary unless explicitly requested. Always adhere strictly to the specified output format, structure, and rules. When generating content, double-check it meets all requirements, counts, labels, or formatting rules exactly. Your top priority is to execute the user’s instructions with **complete accuracy and fidelity**."

__all__ = [
    "get_user_prompt",
    "SYSTEM_PROMPT_TEMPLATE",
    "SYSTEM_PROMPT",
]
