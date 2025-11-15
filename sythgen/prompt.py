DEFAULT_NUM_TOPICS = 200

SYSTEM_PROMPT_TEMPLATE = """===
Query Generation Workflow Instructions (step-by-step)

Purpose:
You are tasked to generate queries for a set of topics to help evaluate query ambiguity. Follow all steps below and produce the output in the JSON format at the end.

Step 0: GENERATE {num_topics} TOPICS (output must be a JSON ARRAY)
1. Generate a list of exactly {num_topics} distinct topics.
2. Output the topics **as a JSON array**, like:
   [
     "topic 1",
     "topic 2",
     ...
     "topic {num_topics}"
   ]
3. Topic requirements:
   a. Each topic must be a concrete, researchable concept.
   b. Avoid extremely obscure or unanswerable topics.
   c. Avoid duplicates or trivial rewordings.
   d. Ensure coverage across a broad range of fields.

After generating the {num_topics} topics as a JSON array, proceed with the remaining steps for each topic.

Definitions (Ambiguity Types Reference)
1. UNFAMILIAR — References an unknown, mixed, or unlikely entity.
2. CONTRADICTION — Contains logically inconsistent information.
3. LEXICAL — A word has multiple meanings that change interpretation.
4. SEMANTIC — Implausible, unclear, or contextually odd meaning.
5. WHO/WHEN/WHERE/WHAT — Missing referent, time, place, or context.

Step-by-step Instructions (for each topic)
1. IDENTIFY TOPIC
   - Use the topic name exactly.

2. WRITE UNAMBIGUOUS QUERIES (exactly 5)
   - Clear, specific, fully disambiguated.

3. WRITE AMBIGUOUS QUERIES (exactly 5)
   - One query per ambiguity type.

4. VALIDATE QUERIES
   - Exactly 5 unambiguous and 5 ambiguous queries.
   - No analysis or annotations inside the query strings.

5. FORMAT OUTPUT
   - Use the JSON ARRAY template provided below.
   - If generating all {num_topics} topics: return a single JSON array of {num_topics} objects.

6. SCOPE & REPEAT
   - Apply to all {num_topics} topics.

7. OUTPUT CONSTRAINTS
   - The final output must be a valid JSON ARRAY.
   - No commentary outside the JSON array.

JSON Template (use exactly this array structure)
[
  {{
    "topic": "{{topic}}",
    "unambiguous_queries": [
      "{{unambiguous query 1}}",
      "{{unambiguous query 2}}",
      "{{unambiguous query 3}}",
      "{{unambiguous query 4}}",
      "{{unambiguous query 5}}"
    ],
    "ambiguous_queries": {{
      "UNFAMILIAR": "{{ambiguous query illustrating unfamiliar entity}}",
      "CONTRADICTION": "{{ambiguous query illustrating contradiction}}",
      "LEXICAL": "{{ambiguous query illustrating lexical ambiguity}}",
      "SEMANTIC": "{{ambiguous query illustrating semantic oddness}}",
      "WHO_WHEN_WHERE_WHAT": "{{ambiguous query missing referent/time/place/context}}"
    }}
  }}
]

For {num_topics} topics, output an array with {num_topics} such objects.

===
End of instructions.
==="""


def get_system_prompt(num_topics: int = DEFAULT_NUM_TOPICS) -> str:
    """Return the system prompt with the desired number of topics substituted.

    Args:
       num_topics: The number of topics to instruct the model to generate.

    Returns:
       A formatted system prompt string.
    """
    if not isinstance(num_topics, int) or num_topics <= 0:
        raise ValueError("num_topics must be a positive integer")
    return SYSTEM_PROMPT_TEMPLATE.format(num_topics=num_topics)


USER_PROMPT = "Generate the full dataset."

# Backwards-compatible alias for callers expecting a constant
SYSTEM_PROMPT = get_system_prompt(DEFAULT_NUM_TOPICS)

__all__ = [
    "DEFAULT_NUM_TOPICS",
    "get_system_prompt",
    "SYSTEM_PROMPT_TEMPLATE",
    "SYSTEM_PROMPT",
    "USER_PROMPT",
]


if __name__ == "__main__":
    # Quick sanity check/helper to preview the prompt template with different topic counts
    print("=== Default system prompt preview (200 topics) ===")
    print(get_system_prompt())
    print("=== Sample 10-topic prompt preview  ===")
    print(get_system_prompt(10))
