# Site-wide dark mode

## Intent
Add a dark-mode toggle throughout the existing Jekyll academic website while preserving its typography, content, and layout.

## Approaches
The user selected a sun/moon button with automatic system preference detection and a remembered manual selection. A three-option System/Light/Dark control was considered but adds unnecessary header complexity. Updating the entire upstream theme would involve unrelated changes.

## Behavior and presentation
Place an accessible, keyboard-operable sun/moon button beside the shared navigation, outside the greedy-navigation overflow mechanism so it stays available on mobile. Follow system preference until a visitor chooses light or dark; remember that choice across pages and visits using localStorage. Apply the theme before the stylesheet paints to avoid a light flash. Storage failure must not prevent switching. Follow system preference changes when there is no saved override.

## Scope and constraints
Use the existing shared Jekyll layout and Sass pipeline, with small dependency-free JavaScript. Keep the existing light appearance. Add charcoal surfaces, soft white text, and readable red links for dark mode, including navigation, sidebar, footer, tables, forms, notices, and code blocks. Preserve photos and figures rather than inverting them. Embedded external documents and third-party frames retain their own appearance. Print output should stay readable on white paper.

## Validation
Build Jekyll if the local toolchain permits it. Verify initial system preference, saved preference, switching, blocked storage, and responsive navigation. Inspect representative pages at desktop and mobile widths when browser tooling is available. No deployment is included.

## Implementation choices
The builder will choose exact accessible colors, icon markup, and theme selectors after inspecting existing component styles. No new frontend framework or runtime dependency is needed.
