# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Personal Jekyll site published to GitHub Pages at **ssaito.net** (`CNAME`). Stack is pinned by the `github-pages` gem: **Jekyll 3.10**, **kramdown 2.4**, **Minimal Mistakes** theme pulled at build time via `remote_theme: mmistakes/minimal-mistakes` (not a local gem). There is no CI — GitHub Pages builds and deploys automatically on push to `master`.

## Preview & deploy

- **Local preview:** `bundle exec jekyll serve` -> http://localhost:4000. First build takes ~100s (fetches the remote theme); later rebuilds are fast. `_config.yml` is **not** hot-reloaded — restart the server after editing it.
- **Deploy:** commit and push to `master`. GitHub Pages rebuilds the live site automatically; no PR or separate deploy step.
- `_config.yml.save` is a stale Minimal Mistakes starter backup (gitignored) — ignore it; `_config.yml` is authoritative.

## Notes workflow (Obsidian -> Jekyll)

Notes are authored in Obsidian and converted with `obsidian_to_jekyll.py` (stdlib-only Python 3):

```
python obsidian_to_jekyll.py <input.md> [output.md]   # stdout if output omitted; warnings go to stderr
```

Placement convention: each note is a folder `_notes/<slug>/index.md` with frontmatter `title` + explicit `permalink: /notes/<slug>/`. Images/PDFs live alongside `index.md` in that folder (the converter emits `_resources/` paths and warns you to copy the files manually — copy them into the note folder). Layout and `mathjax: true` come from the `notes` defaults in `_config.yml`, so notes don't repeat them. After adding a note, link it from `_pages/notes.md` (the manual TOC).

## Math & tables — the fragile part (read before editing any note/post)

kramdown parses emphasis and tables *before* MathJax runs, so **author inline math as `\\(…\\)`, not `$…$`**, in committed `.md` — especially anything containing `_`, `*`, or `|`:

- bare `|` inside inline math splits the line into kramdown **table columns** and breaks rendering -> write `\mid` instead.
- bare `*` -> `\ast`; `}_` -> `}\_` (otherwise eaten as bold/italic markup).
- Display math `$$…$$` goes on its own line with a blank line before and after.
- Tables need a blank line before and after the block to be recognized; math inside table cells must use `\\(…\\)`.

`obsidian_to_jekyll.py` applies all of these fixes automatically — prefer running edits through it. MathJax is loaded manually in `_includes/head/custom.html`. (This is what commits `3a5fc33` / `f0cedc0` fixed.)

## Collections & structure

- `_notes/` and `_projects/` are output-enabled collections (`_projects/` is currently empty; projects live as a static list in `index.md`).
- `_posts/` uses standard `YYYY-MM-DD-title.md` with `categories:`/`tags:` (lowercase-hyphenated, matching the converter's slugify output).
- Debug artifacts at the repo root (`.playwright-mcp/`, `anki_audio_out/`, various `*.png` from table/math debugging) are **not** site content — don't commit them.
