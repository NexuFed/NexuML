# Publish an update

Copy `templates/changelog.mdx` to `src/content/changelog/YYYY-MM-DD-short-slug.mdx`.
Set a real ISO calendar date matching the filename, nonempty title/summary, and
category: `release`, `feature`, `improvement`, or `fix`.

Release announcements require `version: vX.Y.Z` and an HTTPS link to the matching
GitHub Release or tag. Ordinary updates do not need a version. Entries sort newest
first, then ascending filename for equal dates. Their date/slug URLs are permanent;
moving a published entry requires a compatibility redirect.

`links` contain `{label, url}`. Media can be:

```yaml
media:
  - type: image
    src: /changelog/media/example.webp
    alt: Explain what the screenshot demonstrates
  - type: video
    src: /changelog/media/example.webm
    label: Demonstration of the change
    transcript: A useful text alternative to the demonstration.
    # Alternatively provide captions: {src: /changelog/media/example.vtt, lang: en, label: English}
```

Commit local assets under `public/global/changelog/media/`. HTTPS assets are also
accepted. Videos have controls, never autoplay. Keep screenshots compressed and
provide meaningful image alt text and video captions or a text alternative.
Ordinary MDX components and code work in the article body. Use Markdown root-like
links for mount-independent site URLs; do not encode the hosting prefix.

```bash
npm --prefix docs run build:changelog
npm --prefix docs run check
```

Preview the built global target or the assembled local site. Merge through review
to main: validated main publication updates the global changelog automatically,
without editing an index/sidebar or creating a release. Tag publication never
copies its older editorial entries over this timeline.

GitHub Release note categories/exclusions and publishing remain independent and
unchanged. Editorial prose is reviewed by people, not generated from a raw commit
feed. Templates and media-test fixtures are excluded from published collections.
The initial entry is curated from the verified GitHub release published on its
recorded date, not an announcement that this migration has already shipped.
