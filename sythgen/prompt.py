from .topics import topics

USER_PROMPT_TEMPLATE = """
INSTRUCTIONS – FOLLOW EACH STEP PRECISELY:

1. For the topic: “{topic}”, find exactly 5 distinct user intents—each a unique actionable or knowledge-seeking goal.
2. For EACH intent:
   a) Write one **specific query**: clear, precise, directly answering the intent. Assign binary_label = 0.
   b) Write one **ambiguous query**: vague, imprecise, or potentially interpreted in multiple ways. Assign binary_label = 1.
3. Output format (MANDATORY): `topic<TAB>query<TAB>binary_label<TAB>user_information_need`

**STRICT OUTPUT REQUIREMENTS:**
- Output exactly 10 rows per topic: 2 per each intent (one specific, one ambiguous).
- NO headers, notes, or explanations.
- Rows must be separated by line breaks and only use TAB as separator (no extra spaces or blank lines).
- Each specific query MUST have binary_label = 0. Each ambiguous query MUST have binary_label = 1.
- Make sure each intent appears twice (specific and ambiguous) in the list. All queries must be relevant and grammatically correct.
- If you generate more than 10 rows, REMOVE the extras so there are EXACTLY 10.

**FINAL OUTPUT:**
- 10 lines, each following the exact tab-delimited format: topic TAB query TAB binary_label TAB user_information_need

**EXample**
climate change    What are the main causes of climate change?    0    Understand contributing factors to climate change.
climate change    What about climate change?    1    Seeking information about climate change but intent is unclear.
climate change    How does climate change affect weather patterns?    0    Learn about the impact of climate change on weather.
climate change    Does climate change make things different?    1    Looking for effects of climate change but vague on what kind.
climate change    What policies address climate change internationally?    0    Find out about international climate change policies.
climate change    What is being done globally?    1    Want to know about global actions, but it's not specific to climate change.
climate change    What are individual actions to combat climate change?    0    Learn about actions individuals can take.
climate change    How can people help?    1    Looking for ways for individuals to help but not mentioning climate change directly.
climate change    How is climate change measured over time?    0    Understand methodologies for tracking climate change.
climate change    How do they track things over the years?    1    Seeking information on tracking something over time, but it's unclear what.
"""


def get_user_prompt(topic: str) -> str:
    return USER_PROMPT_TEMPLATE.format(topic=topic)


SYSTEM_PROMPT = "You are an extremely precise and obedient assistant. Follow **exactly** what the user instructs, step by step, without adding, omitting, or altering instructions. Do not provide explanations, assumptions, or extra commentary unless explicitly requested. Always adhere strictly to the specified output format, structure, and rules. When generating content, double-check it meets all requirements, counts, labels, or formatting rules exactly. Your top priority is to execute the user’s instructions with **complete accuracy and fidelity**."

__all__ = [
    "get_user_prompt",
    "SYSTEM_PROMPT",
]
