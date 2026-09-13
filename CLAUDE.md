# Project Instructions

Read PROJECT_SPEC.md before any task. It is the build contract, derived from an
approved IGNOU MCA synopsis. The synopsis cannot be changed. Code conforms to it.

## Rule 1 — Synopsis fidelity

Nothing is added that the approved synopsis does not support. Before implementing
anything not already in PROJECT_SPEC.md, stop and state which section of the spec
authorises it. If none does, say so instead of building it.

Explicitly out of scope: authentication, user registration, admin screens,
rating submission, external movie APIs, poster images, JavaScript frameworks,
matrix factorisation, deep learning, Docker.

## Rule 2 — Explainability

The author must defend every file in an oral examination. Prefer the plain
construction over the clever one. No abstraction that is not earning its place.
Every module maps to a numbered module in PROJECT_SPEC.md section 4.

## Rule 3 — Code style

Write as a working engineer would, not as a code generator would.

- No comment that restates the line below it
- No banner comments, no section dividers, no emoji
- Comments only for non-obvious intent: algorithm rationale, dataset quirks,
  deliberate trade-offs
- Type hints on public functions, docstrings on public interfaces only
- No dead code, no commented-out blocks, no unused imports, no print statements

## Rule 4 — Architecture

Layers, strict one-way dependency:

  Presentation -> Application -> Service -> ML -> Data

- Routes hold no business logic and no direct data access
- Services do not import Flask request or response objects
- The ML layer is framework-agnostic and testable standalone
- All persistence goes through repositories
- Configuration is read once, from environment, in app/config.py
- Paths resolve from pathlib, never absolute

## Rule 5 — Working agreement

Build one module at a time. Stop after each and report what changed.
Do not run git commands. The author commits.
Do not create files outside the structure in PROJECT_SPEC.md section 6.
