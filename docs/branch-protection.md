# Branch protection (main)

Configured on GitHub under Settings > Branches > Branch protection rules:

- Require a pull request before merging (no direct pushes to `main`)
- Require at least 1 approving review before merge
- Require branches to be up to date before merging
- Applies to: `main`

Feature work happens on `data-pipeline` and `experiment-tracking`, each opened
as its own PR against `main` and reviewed by the other teammate before merge.

(Note: these rules are configured in the GitHub repo settings UI itself, not
as a file in this repo — this doc is just so the grader/reviewer can see what
was set without needing repo admin access.)
