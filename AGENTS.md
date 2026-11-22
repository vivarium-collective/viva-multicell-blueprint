# AGENTS.md

Guidance for AI coding assistants (and humans following the same flow)
working in this workspace. Read this before opening a PR.

## What this workspace is

`multicell-blueprint` is a viva-superpowers workspace. The Python package
lives under `viva_multicell_blueprint/`; the investigation lives under
`workspace/investigations/<slug>/` and its studies under `workspace/studies/<slug>/`.
See [`README.md`](README.md) for the purpose and [`NEXT_STEPS.md`](NEXT_STEPS.md)
for a tour.

## PR conventions

Two PR types live here.

**Feature / fix PRs** are the normal merge candidates: `gh pr create`, a
conventional-commit title (`feat(...)`, `fix(...)`, `chore(...)`, `docs(...)`), opened
ready-for-review against `main`, CI green before merge. They carry reusable
infrastructure: composites, processes, visualizations, schema bumps, docs.

**Investigation PRs** host a long-running investigation branch and are **not merge
targets**. Title them `investigation: <slug> — <one-line research question>` and open
them with `gh pr create --draft`. The body should say the branch is not a merge target,
link the companion feature PRs, and state the research question and current verdict
(`executive.verdict` / `report.verdict`).

If a change is reusable across investigations, it belongs in a feature PR, not on the
investigation branch.

## Close-investigation workflow

When an investigation is done (research question answered, narrative
written, report rendered), run:

    /viva:investigate close <slug>

This stamps the investigation YAML with `status: closed`, populates
`contributors[]` (derived from `git log` + agent sessions), copies the
rendered HTML report into `workspace/investigations/<slug>/report.html` (git-
tracked artifact), commits on the branch, and opens a PR. **Never auto-
merges.** Reviewer clicks merge in the GitHub UI.

See `/viva:investigate close --help` for details.
