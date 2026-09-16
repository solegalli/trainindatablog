# Writing and editing blog posts

Guidance for writing new posts and editing existing ones in `_posts/`. This file captures conventions learned from working with the author; follow it unless the author says otherwise for a specific post.

## Audience

The reader is a data scientist early in their career. Do not assume they know any concept mentioned in the post. Explain it instead, even if it feels basic.

## Length

Target 3000 to 4000 words for the article body (excluding front matter and code). Going longer than 4000 is fine when the extra length comes from code examples, not from prose. Or when the author specifically asks for it.

## Voice

Before writing or heavily editing a post, read a few existing articles by Sole Galli (`author: sole`) or Cainã Max Couto da Silva (`author: cmcouto`) to calibrate tone and voice. For example:

- `_posts/2020-07-12-feature-engineering-for-machine-learning.md` (sole)
- `_posts/2022-08-12-mutual-information-with-python.md` (sole)
- `_posts/2024-09-20-probability-calibration-in-machine-learning.md` (cmcouto)

Posts by other authors may follow a different voice; do not use them as a style reference unless told to.

## Paragraphs and headings

- Keep paragraphs to 3 sentences or fewer. When a paragraph runs longer, split it into more paragraphs rather than trimming content.
- Use headings generously. More `##`/`###` breaks improve readability, even for sections that feel short.

## Sentence style

- Do not use negative-contrast sentences of the style "It is not A, it's B" or "This is not A. It's B." State the positive claim directly instead.
- Do not use dashes (em dash or en dash). Replace with a comma, a period, or restructure the sentence.

## Tables

- Write tables in standard markdown, then add Bootstrap classes via kramdown's inline attribute list on the line immediately after the table (no blank line in between): `{: .table .table-bordered .table-sm}`. This adds visible borders between all cells and compact padding.
- For a dense table, or one placed in a text-heavy section, also add a smaller font size in the same attribute list, e.g. `{: .table .table-bordered .table-sm style="font-size: 0.85rem;"}`.
- Do not write raw `<table>` HTML. The kramdown attribute list keeps the table maintainable as markdown, consistent with every other table on the site.
- Keep cell text short. If a cell needs a clarifying phrase, use a comma, not a dash (see Sentence style).

## Links

Every post should include:

- At least 3 links to other posts on this blog.
- At least 1 link to one of our products (a book or a course).
- At least 2 external links (research papers, official documentation).

None of the external links should point to an article that competes with this blog on the Google search results page for the post's target keywords. Before adding an external link, check that it is a primary source, official documentation, or otherwise non-competing content, not a rival blog post covering the same topic.

## Editing workflow: re-read after every edit

After making a requested edit, re-read the whole article and check whether the edit requires changes elsewhere. In particular:

- If a section moved, check whether the outline or bullet list near the top of the post (the "This article will explore the following" list) still matches the actual order of sections.
- If a paragraph moved, check for phrases like "as discussed above," "we'll cover this later," or "in the next section" elsewhere in the post. Update or remove references that no longer point to the right place.
- Confirm headings and cross-references are still internally consistent before considering the edit done.

## Keyword coverage

When the author supplies a keyword list, or asks you to research keywords (e.g. by reviewing competitor articles), verify coverage after writing or editing:

- Count occurrences of each keyword in the final text.
- Make sure edits, especially cuts, don't drop any keyword to zero occurrences, and don't thin out keywords that were explicitly prioritized (e.g. the top N of a ranked list) too far.
- Keep keyword use natural. Do not insert a keyword just to hit a count if it reads awkwardly in context.
