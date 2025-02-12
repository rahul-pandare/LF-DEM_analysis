#!/bin/bash

git add .                        # Stages all new and modified files
git commit -m "updated files"    # Commits the changes with a fixed message
git push                         # Pushes changes to the master branch

echo "✅ Changes successfully pushed to GitHub!"  # Displays a success message