# Execution ledger — plan: docs/superpowers/plans/2026-11-21-dark-mode.md

Ruling: Work in the user-requested checkout and leave changes uncommitted, as the approved implementation plan specifies; no branch integration or deployment.
Pre-flight: Task 1 produces html[data-theme], button[data-theme-toggle]; Task 2 consumes those exact selectors. No conflict.
Task 1: complete — 10 behavioral tests failed before implementation and pass afterward; JavaScript syntax check passes.
Task 2: complete — Sass compilation and Jekyll preview build pass. Jekyll dependencies installed into /private/tmp with a temporary Gemfile/lockfile, preserving repository dependencies.
Review: Independent reviewer identified late header measurement, contradictory toggle announcement, and sidebar contrast issues. Reserved control width before greedy-nav measurement; used stable Dark mode accessible name with pressed state and changing title; improved marker/Follow contrast.
Ruling: Standalone _layouts/cv.html is unused by current content; active CV pages use archive/default. The raw talk map is an embedded visualization and retains its own appearance, as specified for frames. Verification will check generated standard routes.

Verification: 10/10 Node behavior tests; node --check; Sass compilation; Jekyll build with _config.yml,_config.dev.yml; generated HTML checks across all 38 pages with shared navigation; git diff --check. Browser: desktop dark homepage, 400px Publications layout, click and Space activation, reload and cross-page persistence. Menu/toggle separation visibly confirmed; mobile menu interaction was not independently verified because native browser control became unavailable.
Build warnings: existing GitHub metadata authentication warning and Liquid syntax warnings for node Sqrt/node Log in _posts/2022-02-09-TensorFlow-Fundamentals.md.
Final independent review: no remaining important defects. Reviewer confirmed compiled CSS reservation and sidebar fixes, generated control coverage, and 10 passing tests. Text, muted text, and link colors each exceed 4.5:1 contrast on both dark surfaces. Changes left uncommitted as planned.
