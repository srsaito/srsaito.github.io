---
name: preview
description: Start a local Jekyll preview server for this site and report the URL. Use when the user wants to preview/serve the site locally, see changes in a browser, or check rendering before pushing.
---

# preview

Start a local Jekyll preview of this Minimal Mistakes / GitHub Pages site.

1. Confirm the toolchain is ready: `bundle check` (run `bundle install` only if it reports missing gems).
2. Start the server in the background, logging to `$CLAUDE_JOB_DIR/tmp/jekyll.log` (or a temp file if that dir is unset):
   `bundle exec jekyll serve --livereload --port 4000`
3. Poll the log until it prints `Server address:` (the first build fetches the remote theme and takes ~100s — be patient, don't assume failure).
4. Verify with `curl -sI http://localhost:4000/` and report **http://localhost:4000** to the user.

Notes:
- Live reload refreshes the browser on `.md`/content saves, but **not** on `_config.yml` edits — restart the server for those.
- The server serves the local working copy, including uncommitted changes — it's a true pre-push preview.
- Leave it running; stop it (kill the background job) only when the user asks.
