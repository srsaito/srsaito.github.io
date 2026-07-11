---
name: convert-note
description: Convert an Obsidian markdown note into a Jekyll note under _notes/ using obsidian_to_jekyll.py. Use when the user wants to add or update a note on the site from an Obsidian source file. Pass the Obsidian .md path as the argument.
---

# convert-note

Convert an Obsidian note into a published Jekyll note. Argument: `$ARGUMENTS` = path to the source Obsidian `.md` (ask if not provided).

1. Pick a slug (kebab-case) from the note title/filename. Target folder: `_notes/<slug>/`.
2. Run the converter, capturing stderr (it lists items needing manual review):
   `python obsidian_to_jekyll.py "<input.md>" _notes/<slug>/index.md`
3. Ensure the frontmatter has `title` and an explicit `permalink: /notes/<slug>/`. Layout + `mathjax: true` come from `_config.yml` defaults — do not add them.
4. **Copy images/PDFs** the converter references (it emits `_resources/...` paths and warns but does NOT copy files). Copy them from the source note's folder into `_notes/<slug>/` (or a `_resources/` subfolder matching the emitted paths) and fix the relative paths so they resolve.
5. Surface every stderr warning to the user (unresolved wikilinks, `<!-- REVIEW -->` math markers, anchor links to verify) and address or flag each.
6. Add a link to the new note in `_pages/notes.md` (the manual TOC).
7. Offer to `/preview` and check the math/tables render before pushing.

Math reminder: inline math must be `\\(…\\)` (not `$…$`) and `|` inside math must be `\mid` — the converter handles this, but verify anything it flagged.
