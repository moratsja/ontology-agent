import json

from src.llm.model import complete
from src.graph.tweets import doc_id, format_documents

ISSUE_TYPES = {
    "one_sided_representation",
    "unsupported_generalization",
    "prompt_framing_bias",
}

BIAS_SYSTEM_PROMPT = """Give feedback on the summary. Return only JSON: {"issues": [...]}
Each issue has issue_type, explanation, evidence_tweet_ids, and suggested_correction.
issue_type is one of: one_sided_representation, unsupported_generalization, prompt_framing_bias.
one_sided_representation: the summary stresses one perspective and leaves out another perspective from the posts that would change the picture for this request. A short summary can leave details out. Report an omission only when it distorts the posts.
unsupported_generalization: the summary treats some posts as true for everyone. Say how to qualify the claim. Do not invent a counterclaim.
prompt_framing_bias: the user request assumes something the posts do not show, and the summary states that assumption as fact. A leading request is not itself a problem.
evidence_tweet_ids are the bracket numbers of the posts you used. Use only numbers that appear in the posts. Do not invent ids.
if you detect an issue, always include a suggested correction for the rewrite that fixes the issue.
For an omitted perspective, cite the posts that were left out. For a generalization, cite the posts that show the claim is too broad, or explain why they cannot support it.
Do not rewrite the summary. Do not report the same problem twice.
"""

# asks model what is wrong with the summary
def bias_review(state):
    raw = complete(BIAS_SYSTEM_PROMPT, _user_message(state))
    documents = state.get("documents") or []
    return {"bias_review": _parse(raw, documents)}

# parses the model's response
def _parse(raw, documents):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _failed(raw)
    if not isinstance(data, dict) or not isinstance(data.get("issues"), list):
        return _failed(raw)
    valid_ids = {doc_id(doc, index) for index, doc in enumerate(documents, start=1)}
    issues = []
    for issue in data["issues"]:
        checked = _issue(issue, valid_ids)
        if checked is None:
            return _failed(raw)
        issues.append(checked)
    return {"status": "ok", "issues": issues}

# checks an issue for validity
def _issue(issue, valid_ids):
    if not isinstance(issue, dict):
        return None
    issue_type = issue.get("issue_type")
    explanation = issue.get("explanation")
    correction = issue.get("suggested_correction")
    if not isinstance(correction, str):
        correction = ""
    tweet_ids = issue.get("evidence_tweet_ids")
    if issue_type not in ISSUE_TYPES:
        return None
    if not isinstance(explanation, str) or not explanation.strip():
        return None
    if not isinstance(tweet_ids, list):
        return None
    cited = []
    for tweet_id in tweet_ids:
        text_id = str(tweet_id)
        if text_id not in valid_ids:
            return None
        cited.append(text_id)
    return {
        "issue_type": issue_type,
        "explanation": explanation,
        "evidence_tweet_ids": cited,
        "suggested_correction": correction,
    }

# formats a failed response
def _failed(raw):
    return {"status": "failed", "issues": [], "response": raw}

# formats the bias review for rewriting
def bias_feedback(review):
    issues = (review or {}).get("issues") or []
    if not issues:
        return (review or {}).get("response") or "None."
    blocks = []
    for issue in issues:
        lines = [f"[{issue.get('issue_type')}] {issue.get('explanation')}"]
        if issue.get("suggested_correction"):
            lines.append(issue["suggested_correction"])
        tweet_ids = issue.get("evidence_tweet_ids") or []
        if tweet_ids:
            lines.append("tweets: " + ", ".join(tweet_ids))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

# message to the model
def _user_message(state):
    return "\n".join(
        [
            f"User request: {state['user_prompt']}",
            "",
            "Summary:",
            state.get("summary") or "",
            "",
            "Posts:",
            format_documents(state.get("documents") or []),
        ]
    )
