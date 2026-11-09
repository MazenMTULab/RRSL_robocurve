# Our workflow
Remotes: `upstream` = robocurve/benchmark-tasks, `origin` = PRIVATE MazenMTULab/RRSL_robocurve_private (build here),
`fork` = PUBLIC MazenMTULab/RRSL_robocurve (submission only).

1. `team-tools/new_task.sh <name>`; build scene + scorer; run in the Brev sim; record model run, demo, screenshot.
2. `source .env` (copy .env.example; never commit). `python tools/upload_media.py tasks/<name> --video ... --demo ... --screenshot ...`
3. `BLIND_TERMS=<team words> python3 team-tools/validate_task.py tasks/<name> --strict-media`
4. Commit to `origin`. `team-tools/submit.sh <name>` pushes a clean branch to the fork; `--open-pr` opens the PR.
5. `gh pr view --repo robocurve/benchmark-tasks <n> --comments`; revise by pushing to the same fork branch.
