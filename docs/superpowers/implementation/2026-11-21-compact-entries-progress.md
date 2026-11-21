# Execution ledger — compact entries

Plan: docs/superpowers/plans/2026-11-21-compact-entries.md
Design and direct implementation approved in conversation.

Task 1: complete. Added scoped Liquid includes and Sass; only Publications and Notes list pages opt in. Current thesis receives accurate author/type metadata, with its full citation and resource URLs preserved.
Task 2: verification. Real Jekyll rendering fixture checks cover titles, optional metadata, local/external links, short plain-text excerpts, thumbnails, dates, resource labels and native disclosure controls. Existing 10 theme tests pass. Sass compiles; Jekyll build succeeds with preexisting GitHub auth and TensorFlow Liquid warnings.
Scope check: 49 generated HTML pages unchanged after ignoring inter-tag whitespace; only publications/index.html and notes/index.html have content changes (checked before final link/empty-field fixes, whose scope is the same dedicated include).
Visual checks: desktop dark Publications, desktop light note cards using temporary unpublished fixtures, and 400px-wide light note cards. Existing fonts, palette and page width retained. The user’s live preview at 4001 was temporarily blank, so visual checks used the static preview at 4000.
Independent review found relative-link prefix loss and truthy empty optional values. Added regression checks, observed failure, and corrected both. Also fixed double-escaped ampersands caught by fixture verification. Follow-up review confirmed those fixes and identified a colon-in-query edge case. Replaced custom scheme detection with Jekyll’s built-in absolute_url resolver and added a failing-then-passing regression case for both root-relative and relative DOI query links. All rendering regression checks pass.
Ruling: keep source changes in the user-requested checkout, uncommitted. Temporary note fixtures exist only under /private/tmp; no invented notes added to the repository.
Cursor report remains unresolved pending screenshot; no speculative changes to cursor or accessibility behavior.
