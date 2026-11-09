#!/usr/bin/env bash
# usage: team-tools/submit.sh <task-name> [--open-pr]
# Copies ONE finished task from this private repo into a fresh branch of the public fork, pushes it,
# and (only with --open-pr) opens the PR to robocurve/benchmark-tasks.
set -euo pipefail
cd "$(dirname "$0")/.."
name=${1:?task name}; open=${2:-}
python3 team-tools/validate_task.py "tasks/$name" --strict-media
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git clone --quiet https://github.com/MazenMTULab/RRSL_robocurve.git "$tmp/fork"
cd "$tmp/fork"
git remote add upstream https://github.com/robocurve/benchmark-tasks.git
git fetch --quiet upstream main
git checkout --quiet -b "$name" upstream/main
cp -r "$OLDPWD/tasks/$name" "tasks/$name"
git add "tasks/$name"
git commit --quiet -m "Add $name task"
git push -u origin "$name"
if [[ $open == --open-pr ]]; then
  gh pr create --repo robocurve/benchmark-tasks --head "MazenMTULab:$name" --fill
else
  echo "pushed branch $name to the fork. Review, then rerun with --open-pr"
fi
