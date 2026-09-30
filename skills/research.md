# Research Skill

When the user wants facts looked up or information gathered, follow these steps:

1. **Identify the core question.** What exactly are they asking? Strip out filler.
2. **Choose search terms.** Use specific nouns and entities. Avoid vague words like "stuff" or "thing".
3. **Call the `research(query)` tool** with a clean, focused query — not the raw user message.
4. **Verify the result is relevant.** If the result doesn't match the question, retry with different terms.
5. **Pass findings on** — the user (or another skill) will use them next.

Always load this skill (via `use_skill('research')`) BEFORE calling the `research` tool.