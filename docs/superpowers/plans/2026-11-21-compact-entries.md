# Compact Entries Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan directly in the existing user-requested checkout. Leave changes uncommitted.

**Goal:** Compact publication rows and shallow note cards on their listing pages only.
**Architecture:** Dedicated collection-entry Liquid include, used only by the two listing pages; isolated Sass partial using existing palette and typography.
**Tech Stack:** Jekyll, Liquid, Sass.
**Spec:** docs/superpowers/specs/2026-11-21-compact-entries-design.md

## Global Constraints

Preserve fonts, colors, page width and all other pages. Reuse existing content and links. No invented published notes, new libraries, or site deployment. Continue the previously selected direct execution method.

## Review Focus

- Long titles and URLs must wrap without overflow.
- Missing optional metadata must not leave empty action rows or separators.
- The current thesis has a nonstandard collection metadata value; explicit include kind determines its rendering.
- Local note links must respect base_path; external resource links must remain intact.
- Dark styling and print must reuse existing palette without altering other archive pages.

### Task 1: Render compact entries

- [x] Create _includes/collection-entry.html with explicit post and kind parameters, accessible resource labels, optional authors/type/venue/date, expandable abstract and citation, and short note excerpt.
- [x] Switch only _pages/publications.md and _pages/notes.html to the include inside scoped list wrappers. Add author/type metadata to the existing thesis.
- [x] Create _sass/_collection-entries.scss, import it at the end of main.scss. Keep inherited font families and existing colors; compact spacing and font sizes; full-width one-column cards; optional small thumbnail beside text; responsive wrapping; screen-only dark styling.

### Task 2: Verify scope and presentation

- [x] Compile Sass and build Jekyll using the existing temporary toolchain.
- [x] Render representative notes and publications with and without optional fields using fixtures outside the production source. Check resource URLs, excerpt escaping, heading semantics, and no empty controls.
- [x] Compare generated non-target pages against a baseline build, inspect light/dark and narrow layouts when browser control permits, run existing theme tests and git diff --check.
- [x] Request an independent review and address material findings. Record results and leave changes uncommitted.
