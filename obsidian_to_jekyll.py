#!/usr/bin/env python3
"""
obsidian_to_jekyll.py

Converts an Obsidian Markdown note to Jekyll (Minimal Mistakes) compatible Markdown.

Transformations applied automatically:
  1. Adds YAML frontmatter (title from filename) if missing
  2. Converts Obsidian callouts  >[!type]  to plain blockquotes
  3. Converts embedded images  ![[file.png]]  to  ![](_resources/file.png)
  4. Converts wikilinks: [[note|alias]] → alias, [[#Heading]] → [Heading](#anchor),
     [[note]] → note text (with warning)
  5. Strips zotero:// links, keeping the label text
  6. Converts inline  $...$  to  \(...\)  so kramdown doesn't mangle _ and * inside math
  7. Ensures blank lines before and after display math $$ blocks
  8. Splits $$formula$$ that is glued to end of a text/list line

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


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def convert(text, title='Untitled'):
    warnings = []

    # Protect fenced code blocks from all transformations
    text, code_blocks = protect_code_blocks(text)

    text = add_frontmatter(text, title)
    text = convert_callouts(text)
    text = convert_embedded_images(text, warnings)
    text = convert_wikilinks(text, warnings)
    text = strip_zotero_links(text)
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
