"""LLM-assisted result summary.

Turns the numeric results into a short plain-English note for non-technical
stakeholders (the "translate model outputs into operational metrics" duty).

* If ANTHROPIC_API_KEY is set and the `anthropic` package is installed, the
  metrics JSON is sent to an LLM (model name from env LLM_MODEL).
* Otherwise a deterministic template is used, so the pipeline always runs
  offline. The template output is what is stored when no key is available.
"""
import json
import os

PROMPT = (
    "You are an ML analyst. In at most 120 words, explain these video-classifier "
    "results to a product manager. State clearly which model is fairest, what "
    "the lighting-condition bias did, and one recommended next step. "
    "Use only the numbers given.\n\nRESULTS JSON:\n{payload}"
)


def _template(m):
    b, r = m["baseline_cnn"], m["reweighed_cnn"]
    return (
        f"The baseline video model reaches {b['overall']*100:.1f}% overall accuracy, but only "
        f"{b['worst_group']*100:.1f}% on its weakest lighting group (gap {b['gap']*100:.1f} points), "
        f"showing it partly relies on background brightness instead of motion. "
        f"After reweighing the training data, worst-group accuracy is {r['worst_group']*100:.1f}% "
        f"(gap {r['gap']*100:.1f} points). Recommended next step: validate on real footage with "
        f"audited lighting metadata before any deployment."
    )


def summarise(metrics: dict) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"),
                max_tokens=400,
                messages=[{"role": "user",
                           "content": PROMPT.format(payload=json.dumps(metrics, indent=2))}])
            return {"source": "llm", "text": msg.content[0].text}
        except Exception as e:                      # network / key problems -> fall back
            return {"source": f"template (LLM failed: {type(e).__name__})", "text": _template(metrics)}
    return {"source": "template (no API key set)", "text": _template(metrics)}
