#!/usr/bin/env bash
# usage: team-tools/new_task.sh <task-name>   (lowercase-with-dashes)
set -euo pipefail
cd "$(dirname "$0")/.."
name=${1:?task name}
[[ $name =~ ^[a-z0-9-]+$ ]] || { echo "use lowercase letters, digits, dashes"; exit 1; }
[[ -e tasks/$name ]] && { echo "tasks/$name exists"; exit 1; }
cp -r templates/task "tasks/$name"
echo "created tasks/$name; add scene + scorer code, fill task.json and README.md"
