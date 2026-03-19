# ADR-005: Prompt Versioning and Regression Testing

## Status
Accepted

## Context
Prompt changes can silently degrade agent behavior. A small wording change that improves one classification can break another. We need version control and regression testing for prompts.

## Decision
Treat prompts as versioned artifacts stored alongside code.

### Storage
```
data/prompts/
├── v1/
│   ├── classify_email.txt
│   ├── extract_metadata.txt
│   └── search_response.txt
└── v2/
    └── classify_email.txt
```

### Loading
The `PromptRegistry` loads prompts by name and version:
```python
registry = PromptRegistry()
prompt = registry.get_prompt("classify_email", version=1, subject=email.subject)
```

### Versioning Rules
1. Every prompt change creates a new version directory
2. Old versions are never modified — they're immutable
3. Each tenant can be pinned to a specific version (for gradual rollout)
4. The CI pipeline runs both versions against the eval set before merging

### Regression Testing
The test suite (`test_prompt_registry.py`) loads each version and runs it against a set of test cases:
- 50+ test emails with known correct classifications
- Each prompt version must match or exceed the previous version's accuracy
- If accuracy drops on any category, the PR is blocked

## Why Not Just Store Prompts in Code?
- Prompts change more frequently than code logic
- Non-engineers (product managers, domain experts) may need to review/suggest prompt changes
- Version comparison is easier with file-based prompts than git diffing embedded strings
- In production, prompts can be stored in S3 with versioning enabled

## Consequences
- Prompt changes require updating test expectations if behavior intentionally changes
- Need a golden eval dataset that's maintained as the domain evolves
- Deployment must pin a specific prompt version — no "latest" in production
