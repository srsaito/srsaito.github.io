#!/usr/bin/env python3
"""
obsidian_to_jekyll.py

Converts an Obsidian Markdown note to Jekyll (Minimal Mistakes) compatible Markdown.

Transformations applied automatically:
  1. Adds YAML frontmatter (title from filename) if missing
  2. Normalizes frontmatter: splits 'Category/Sub' on '/' → category + tag,
     lowercases and hyphenates all category/tag values
  3. Converts Obsidian callouts  >[!type]  to plain blockquotes
  4. Converts embedded images  ![[file.png]]  to  ![](_resources/file.png)
  5. Converts wikilinks: [[note|alias]] → alias, [[#Heading]] → [Heading](#anchor),
     [[note]] → note text (with warning)
  6. Strips zotero:// links, keeping the label text
  7. Converts inline  $...$  to  \(...\)  so kramdown doesn't mangle _ and * inside math
  8. Ensures blank lines before and after display math $$ blocks
  9. Splits $$formula$$ that is glued to end of a text/list line

Items flagged for manual review (marked with # REVIEW in output):
  - $$ block opening embedded mid-line (e.g. inside a blockquote)
  - Any $$ pattern not cleanly resolved

Usage:
  python obsidian_to_jekyll.py <input.md> [output.md]

  If output.md is omitted, result is printed to stdout.
  Warnings are printed to stderr.

Image note:
  Embedded images (![[...]]) are converted to standard Markdown but the image
  files themselves must be copied manually to the same directory as the output note.
"""

import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Code-block protection — prevents transformations inside ``` fences
# ---------------------------------------------------------------------------

def protect_code_blocks(text):
    blocks = {}
    counter = [0]

    def replacer(m):
        key = f'\x00CODE{counter[0]}\x00'
        blocks[key] = m.group(0)
        counter[0] += 1
        return key

    protected = re.sub(r'```[\s\S]*?```', replacer, text)
    return protected, blocks


def restore_code_blocks(text, blocks):
    for key, value in blocks.items():
        text = text.replace(key, value)
    return text


# ---------------------------------------------------------------------------
# Individual transformations
# ---------------------------------------------------------------------------

def add_frontmatter(text, title):
    """Prepend YAML frontmatter if the note has none."""
    if text.lstrip().startswith('---'):
        return text
    return f'---\ntitle: "{title}"\n---\n\n' + text


def convert_callouts(text):
    """>[!type] optional title  →  >  (plain blockquote, type line stripped)."""
    return re.sub(r'^([ \t]*)>\s*\[![^\]]+\][^\n]*', r'\1>', text, flags=re.MULTILINE)


def convert_embedded_images(text, warnings):
    """![[image.png]]  →  ![](_resources/image.png)

    Obsidian stores attachments in a _resources/ folder next to the note.
    The converted path keeps that relative reference so images can be copied
    from <note_dir>/_resources/ to <output_dir>/_resources/.
    """
    images = re.findall(r'!\[\[([^\]]+)\]\]', text)
    if images:
        warnings.append(
            "Copy these files from <note_dir>/_resources/ to <output_dir>/_resources/:\n"
            + '\n'.join(f"       {img}" for img in images)
        )
    return re.sub(r'!\[\[([^\]]+)\]\]', lambda m: f'![](_resources/{m.group(1)})', text)


def convert_wikilinks(text, warnings):
    """
    [[note|alias]]  → alias
    [[#Heading]]    → [Heading](#heading-anchor)
    [[note]]        → note  (plain text, with warning)
    """
    def replace(m):
        inner = m.group(1)
        if '|' in inner:
            return inner.split('|', 1)[1]
        if inner.startswith('#'):
            heading = inner[1:]
            anchor = re.sub(r'[^a-zA-Z0-9\s-]', '', heading).strip().lower().replace(' ', '-')
            warnings.append(
                f"Wikilink [[{inner}]] → [{heading}](#{anchor}) — verify the anchor slug"
            )
            return f'[{heading}](#{anchor})'
        warnings.append(f"Wikilink [[{inner}]] → plain text '{inner}'")
        return inner

    return re.sub(r'\[\[([^\]]+)\]\]', replace, text)


def ensure_blank_lines_around_tables(text):
    """
    Kramdown requires a blank line before a table to recognize it as a table block.
    Without it, a table immediately after a heading or paragraph is rendered as <p>.
    """
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Detect a table row (starts with |)
        if stripped.startswith('|'):
            prev = result[-1] if result else ''
            if prev.strip():
                result.append('')
        result.append(line)
        # Ensure blank line after table (when next line is not a table row)
        if stripped.startswith('|'):
            next_line = lines[i + 1] if i + 1 < len(lines) else ''
            if next_line.strip() and not next_line.strip().startswith('|'):
                result.append('')
    return '\n'.join(result)


def strip_zotero_links(text):
    """[label](zotero://...)  →  label"""
    return re.sub(r'\[([^\]]*)\]\(zotero://[^)]*\)', r'\1', text)


def convert_inline_math_delimiters(text):
    """
    Convert inline  $...$  to  \\(...\\).

    kramdown processes markdown emphasis (_italic_, *bold*) before passing
    content to MathJax, so  $q_*(a^*)$  gets mangled — underscores and
    asterisks inside $...$ are treated as markup.  Using \\(...\\) delimiters
    sidesteps this entirely because kramdown leaves them untouched.

    Skips  $$...$$  display math (already handled separately).
    """
    result = []
    i = 0
    while i < len(text):
        # Skip $$ display math — advance past the whole block
        if text[i:i+2] == '$$':
            end = text.find('$$', i + 2)
            if end == -1:
                result.append(text[i:])
                break
            result.append(text[i:end + 2])
            i = end + 2
            continue

        # Single $ — possible inline math
        if text[i] == '$':
            end = text.find('$', i + 1)
            # Make sure it's not a $$ closer
            while end != -1 and text[end:end+2] == '$$':
                end = text.find('$', end + 2)
            if end == -1:
                result.append(text[i:])
                break
            inner = text[i+1:end]
            # Only convert if it looks like math (contains LaTeX-ish content)
            if inner and '\n' not in inner:
                # Replace bare * with \ast so kramdown doesn't treat it as
                # emphasis markup. \ast renders identically in LaTeX.
                inner = re.sub(r'(?<!\\)\*', r'\\ast', inner)
                # Replace }_ with }\_ so kramdown treats _ as an escaped
                # literal (not emphasis), while MathJax still sees it as _.
                # This fixes e.g. \bar{o}_{n} where } precedes _ triggering
                # kramdown's left-flanking emphasis rule.
                inner = inner.replace('}_', '}\\_')
                # Replace bare | with \mid so kramdown doesn't treat it as a
                # table column separator (e.g. E[R_t|A_t] would split lines
                # into table cells, breaking the surrounding \\(\\) delimiters).
                # \mid renders identically in probability/conditional notation.
                inner = re.sub(r'(?<!\\)\|', r'\\mid ', inner)
                # Use \\( ... \\) in the markdown file.
                # kramdown renders  \\(  →  \(  in HTML, which MathJax processes.
                # Plain  \(  is eaten by kramdown as an escaped parenthesis.
                result.append(r'\\(' + inner + r'\\)')
            else:
                result.append(text[i:end+1])
            i = end + 1
            continue

        result.append(text[i])
        i += 1

    return ''.join(result)


def fix_display_math(text, warnings):
    """
    Ensure blank lines before and after every display-math $$ block.

    Handles:
      A. Single-line block  $$formula$$  that starts its line
      B. Multi-line block   $$\\begin{...} … \\end{...}$$
      C. $$formula$$ glued to end of a text or list-item line  → split off

    Flags (with # REVIEW comment) any remaining $$ that couldn't be resolved.
    """
    lines = text.split('\n')
    result = []
    in_math = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        next_stripped = lines[i + 1].strip() if i + 1 < len(lines) else ''

        if not in_math:

            # ----------------------------------------------------------------
            # Case C: $$formula$$ glued to end of text/list line
            # Pattern: something non-$$ then $$content$$ at end of line
            # ----------------------------------------------------------------
            glued = re.match(r'^(.*?[^$\s])\s*\$\$(.+?)\$\$\s*$', line)
            if glued and not stripped.startswith('$$'):
                prefix = glued.group(1).rstrip()
                formula = glued.group(2)
                warnings.append(
                    f"L{i+1}: Display math split from text — review list/blockquote context:\n"
                    f"       {stripped[:80]}"
                )
                if prefix:
                    result.append(prefix)
                # Blank before math block
                if result and result[-1].strip():
                    result.append('')
                result.append(f'$${formula}$$')
                # Blank after
                if next_stripped:
                    result.append('')
                continue

            # ----------------------------------------------------------------
            # Detect $$ embedded mid-line in a way we can't auto-fix
            # (e.g. blockquote text ending with $$\begin{equation})
            # ----------------------------------------------------------------
            if '$$' in stripped and not stripped.startswith('$$'):
                # Check for an unclosed $$ (opens multi-line block mid-line)
                after_last = stripped.rsplit('$$', 1)
                # If there's an odd number of $$ tokens on the line it's unclosed
                if stripped.count('$$') % 2 == 1:
                    warnings.append(
                        f"L{i+1}: Unclosed $$ mid-line — manual fix required:\n"
                        f"       {stripped[:80]}"
                    )
                    result.append(f'<!-- REVIEW: unclosed $$ mid-line -->')
                    result.append(line)
                    in_math = True  # treat subsequent lines as math block
                    continue

            # ----------------------------------------------------------------
            # Cases A & B: line starts a display math block (starts with $$)
            # ----------------------------------------------------------------
            math_start = re.match(r'^(\s*>?\s*)\$\$', line)
            if math_start:
                # Ensure blank line before (skip if already blank or right after heading)
                prev = result[-1] if result else ''
                if prev.strip() and not prev.strip().startswith('#'):
                    result.append('')

                is_single_line = (
                    stripped.endswith('$$')
                    and stripped != '$$'
                    and stripped.count('$$') >= 2
                    and len(stripped) > 4
                )

                if is_single_line:
                    # Case A: $$formula$$  on its own line
                    result.append(line)
                    if next_stripped:
                        result.append('')
                else:
                    # Case B: opening of a multi-line block
                    in_math = True
                    result.append(line)

                continue

            result.append(line)

        else:
            # ----------------------------------------------------------------
            # Inside a multi-line math block — pass through until closing $$
            # ----------------------------------------------------------------
            result.append(line)
            if stripped.endswith('$$'):
                in_math = False
                if next_stripped:
                    result.append('')

    return '\n'.join(result)


def normalize_frontmatter(text, warnings):
    """Normalize Obsidian-style frontmatter for Jekyll compatibility.

    Splits category paths on '/':
      - First segment → lowercased, hyphenated category
      - Remaining segments → appended to 'tags' list
    Also slugifies all tag values (lowercase, spaces to hyphens).
    """
    if not text.lstrip().startswith('---'):
        return text

    fm_end = text.find('\n---', 4)
    if fm_end == -1:
        return text

    after_close = text.find('\n', fm_end + 1)
    body = text[after_close + 1:] if after_close != -1 else ''
    fm_content = text[4:fm_end]

    def slugify(s):
        return re.sub(r'\s+', '-', s.strip().lower())

    lines = fm_content.split('\n')
    out = []
    extra_tags = []
    existing_tags = []
    in_cats = False
    in_tags = False
    tags_header_idx = None

    for line in lines:
        stripped = line.strip()

        if not stripped:
            out.append(line)
            continue

        # Top-level YAML key (not indented)
        if not line[0].isspace():
            key_m = re.match(r'^([\w][\w ]*):\s*(.*)', line)
            if key_m:
                key = key_m.group(1).strip()
                inline_val = key_m.group(2).strip()

                if key == 'categories':
                    in_cats, in_tags = True, False
                    if inline_val.startswith('[') and inline_val.endswith(']'):
                        items = [v.strip().strip('"\'') for v in inline_val[1:-1].split(',') if v.strip()]
                        new_cats = []
                        for item in items:
                            parts = [slugify(p) for p in item.split('/')]
                            new_cats.append(parts[0])
                            extra_tags.extend(parts[1:])
                        out.append(f'categories: [{", ".join(new_cats)}]')
                        in_cats = False
                    else:
                        out.append(line)
                    continue

                if key == 'tags':
                    in_cats, in_tags = False, True
                    tags_header_idx = len(out)
                    if inline_val.startswith('[') and inline_val.endswith(']'):
                        items = [v.strip().strip('"\'') for v in inline_val[1:-1].split(',') if v.strip()]
                        existing_tags.extend(slugify(v) for v in items)
                        in_tags = False
                    out.append('tags:')
                    continue

                in_cats, in_tags = False, False

        # List item under categories or tags
        if stripped.startswith('-') and (in_cats or in_tags):
            val = stripped[1:].strip().strip('"\'')
            if in_cats:
                parts = [slugify(p) for p in val.split('/')]
                out.append(f'  - {parts[0]}')
                extra_tags.extend(parts[1:])
            else:
                existing_tags.append(slugify(val))
            continue

        out.append(line)

    # Rebuild tags section with merged values
    all_tags = existing_tags + extra_tags
    if tags_header_idx is not None:
        tag_items = [f'  - {t}' for t in all_tags]
        out = out[:tags_header_idx + 1] + tag_items + out[tags_header_idx + 1:]
    elif extra_tags:
        out.append('tags:')
        out.extend(f'  - {t}' for t in extra_tags)

    if extra_tags:
        warnings.append(f"Category sub-path(s) moved to tags: {extra_tags}")

    fm_out = '\n'.join(out)
    return f'---\n{fm_out}\n---\n{body}'


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def convert(text, title='Untitled'):
    warnings = []

    # Protect fenced code blocks from all transformations
    text, code_blocks = protect_code_blocks(text)

    text = add_frontmatter(text, title)
    text = normalize_frontmatter(text, warnings)
    text = convert_callouts(text)
    text = convert_embedded_images(text, warnings)
    text = convert_wikilinks(text, warnings)
    text = strip_zotero_links(text)
    text = ensure_blank_lines_around_tables(text)
    text = convert_inline_math_delimiters(text)
    text = fix_display_math(text, warnings)

    text = restore_code_blocks(text, code_blocks)

    return text, warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: '{input_path}' not found", file=sys.stderr)
        sys.exit(1)

    title = input_path.stem
    text = input_path.read_text(encoding='utf-8')

    converted, warnings = convert(text, title)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(converted, encoding='utf-8')
        print(f"Written to: {output_path}")
    else:
        print(converted)

    if warnings:
        print(f"\n{'─' * 60}", file=sys.stderr)
        print(f"WARNINGS — {len(warnings)} item(s) need review:", file=sys.stderr)
        for w in warnings:
            print(f"  • {w}", file=sys.stderr)
        print(f"{'─' * 60}", file=sys.stderr)


if __name__ == '__main__':
    main()
