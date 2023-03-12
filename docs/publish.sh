#! /bin/bash
#cd /docs/build

SRC_FOLDER_PATH: 'docs/build'
TARGET_BRANCH: 'gh-pages'

files=$(find $SRC_FOLDER_PATH -type f) # get the file list
#git config --global user.name 'GitHub Action'
#git config --global user.email 'action@github.com'
git fetch                         # fetch branches
git checkout $TARGET_BRANCH       # checkout to your branch
git checkout ${GITHUB_REF##*/} -- $files # copy files from the source branch
git add -A
git diff-index --quiet HEAD ||  git commit -am "deploy files"  # commit to the repository (ignore if no modification)
git push origin $TARGET_BRANCH # push to remote branch

read -p "Press enter to continue"