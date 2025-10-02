# Moderation

Uses OpenAI's [moderation API](https://platform.openai.com/docs/guides/moderation) to detect harmful or policy-violating content including hate speech, harassment, self-harm, and other inappropriate content. Analyzes text using OpenAI's trained moderation models, flags content that violates OpenAI's usage policies, and provides category-specific violation scores.

## Configuration

```json
{
    "name": "Moderation",
    "config": {
        "categories": ["hate", "violence", "self-harm", "sexual"]
    }
}
```

### Parameters

- **`categories`** (optional): List of content categories to check for violations. If not specified, all categories are checked.

**Available categories:**

- `hate` - Hate speech and discriminatory content
- `hate/threatening` - Hateful content that also includes violence or serious harm
- `harassment` - Harassing or bullying content
- `harassment/threatening` - Harassment content that also includes violence or serious harm
- `self-harm` - Content promoting or depicting self-harm
- `self-harm/intent` - Content where the speaker expresses intent to harm oneself
- `self-harm/instructions` - Content that provides instructions for self-harm
- `violence` - Content that depicts death, violence, or physical injury
- `violence/graphic` - Content that depicts death, violence, or physical injury in graphic detail
- `sexual` - Sexually explicit or suggestive content
- `sexual/minors` - Sexual content that includes individuals under the age of 18
- `illicit` - Content that gives advice or instruction on how to commit illicit acts
- `illicit/violent` - Illicit content that also includes references to violence or procuring a weapon

## Implementation Notes

- **OpenAI API Required**: Uses OpenAI's moderation API therefore requires an OpenAI API key (no cost)
- **Policy-Based**: Follows OpenAI's content policy guidelines

## What It Returns

Returns a `GuardrailResult` with the following `info` dictionary:

```json
{
    "guardrail_name": "Moderation",
    "flagged": true,
    "categories": {
        "hate": true,
        "violence": false,
        "self-harm": false,
        "sexual": false
    },
    "category_scores": {
        "hate": 0.95,
        "violence": 0.12,
        "self-harm": 0.08,
        "sexual": 0.03
    },
    "checked_text": "Original input text"
}
```

- **`flagged`**: Whether any category violation was detected
- **`categories`**: Boolean flags for each category indicating violations
- **`category_scores`**: Confidence scores (0.0 to 1.0) for each category
- **`checked_text`**: Original input text
