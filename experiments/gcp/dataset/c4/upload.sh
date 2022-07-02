#!/bin/bash -x
shopt -s extglob
mv ../bigscience-metadata/outputs/c4-en-html_cc-main-2019-18_pq$1 .
git add c4-en-html_cc-main-2019-18_pq$1
git commit -m "feat: tag metadata on pq${1}"
git push
git lfs prune -f
rm c4-en-html_cc-main-2019-18_pq$1
git reset --hard
