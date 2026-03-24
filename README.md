# Agent Banana

Agent Banana is an end-to-end image editing agent with explicit planning, preview-first localization, region editing, and quality gating.

## What it does

- Parses a natural-language edit request into atomic edit steps.
- Enumerates candidate edit paths at planning time and ranks them with a persisted RL-style value store.
- Generates one Nano Banana preview before each bounding-box decision.
- Infers a region from the source/preview delta, then tightens it with target-aware priors for small accessories like glasses.
- Edits only the selected crop, composites it back into the original image, and rejects oversized or overly destructive local edits.
- Persists session memory and planner feedback across turns.

## Quick start

Install the package:

```bash
python -m pip install -e .
```

Run the browser UI:

```bash
python -m agent_banana.server --host 127.0.0.1 --port 8010
```

Or run the CLI:

```bash
python -m agent_banana.cli \
  --image /path/to/input.png \
  --instruction "Remove only the glasses from the woman's face. Preserve her eyes, skin, hair, and pose."
```

## Environment

If `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set, the app uses Gemini image generation for preview creation and crop editing.

Without an API key, the app still runs end to end in deterministic mock mode so the planner, bbox logic, quality gates, and UI remain testable.

Optional `.env`:

```bash
GEMINI_API_KEY="replace-with-your-gemini-api-key"
AGENT_BANANA_IMAGE_MODEL="gemini-2.5-flash-image"
```

## Project layout

- `src/agent_banana/planning.py`: instruction parsing, path search, and RL-style scoring.
- `src/agent_banana/nano_banana.py`: Gemini Nano Banana client and deterministic mock backend.
- `src/agent_banana/pipeline.py`: orchestration across planning, previewing, localization, editing, compositing, and reward updates.
- `src/agent_banana/targeting.py`: target-aware priors for compact edits such as glasses removal.
- `src/agent_banana/quality.py`: local quality gate with size-aware rejection.
- `src/agent_banana/server.py`: local browser UI.
- `tests/test_agent_banana.py`: regression coverage for planning, localization, and end-to-end flow.
