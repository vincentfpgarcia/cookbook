# Git

## Commands

### Log

| Description | Command |
| --- | --- |
| Show commit history | `git log` |
| Compact log (one line per commit) | `git log --oneline` |
| Log with graph | `git log --oneline --graph` |
| Log by author | `git log --author="name"` |
| Log for a specific file | `git log /path/to/file` |
| Log with diff | `git log -p` |
| Show last N commits | `git log -n 5` |

### Differences

| Description | Command |
| --- | --- |
| Display diff | `git diff` |
| Display diff with HEAD | `git diff HEAD` |
| Display staged diff | `git diff --cached` |

### Staging

| Description | Command |
| --- | --- |
| Stage one file | `git add /path/to/file` |
| Stage all changes (do not use) | `git add .` |
| Stage changes interactively (preferred) | `git add -p` |

### Commit

| Description | Command |
| --- | --- |
| Commit (vi interface) | `git commit` |
| Quick commit | `git commit -m "commit message"` |
| Amend last commit (local) | `git commit --amend` |
| Amend last commit (already pushed) | `git commit --amend` <br> `git push --force-with-lease` |

### Branches

| Description | Command |
| --- | --- |
| List local branches | `git branch` |
| List all branches | `git branch -a` |
| Switch to a branch | `git switch my-branch` |
| Create a branch | `git switch -c my-branch` |
| Push local branch | `git push -u origin my-branch` |
| Remove local branch | `git branch -d my-branch` <br> `git branch -D my-branch` |
| Remove remote branch | `git push origin --delete my-branch` |
| Update the list of branches | `git remote update origin --prune` |

### Remote

| Description | Command |
| --- | --- |
| List remotes | `git remote -v` |
| Add a remote | `git remote add origin url` |
| Change remote URL | `git remote set-url origin url` |
| Remove a remote | `git remote remove origin` |

### Stash

| Description | Command |
| --- | --- |
| Stash current changes | `git stash` |
| Stash with a message | `git stash push -m "message"` |
| List all stashes | `git stash list` |
| Show stash contents | `git stash show` |
| Show stash contents (full diff) | `git stash show -p` |
| Show a specific stash | `git stash show stash@{n}` |
| Show a specific stash (full diff) | `git stash show -p stash@{n}` |
| Apply and remove last stash | `git stash pop` |
| Apply and remove a specific stash | `git stash pop stash@{n}` |
| Remove last stash without applying | `git stash drop` |
| Remove a specific stash | `git stash drop stash@{n}` |
| Remove all stashes | `git stash clear` |

### Blame

| Description | Command |
| --- | --- |
| Show who changed each line | `git blame /path/to/file` |
| Blame a specific range of lines | `git blame -L 10,20 /path/to/file` |
| Ignore whitespace changes | `git blame -w /path/to/file` |

### Tags

| Description | Command |
| --- | --- |
| List tags | `git tag` |
| Create a lightweight tag | `git tag v1.0.0` |
| Create an annotated tag | `git tag -a v1.0.0 -m "message"` |
| Tag a specific commit | `git tag v1.0.0 123456` |
| Push a tag to remote | `git push origin v1.0.0` |
| Push all tags to remote | `git push origin --tags` |
| Remove a local tag | `git tag -d v1.0.0` |
| Remove a remote tag | `git push origin --delete v1.0.0` |

### Undo

| Description | Command |
| --- | --- |
| Discard local changes of one file | `git restore /path/to/file` |
| Discard changes interactively | `git restore -p` |
| Reset last commit and keep changes | `git reset --soft HEAD~1` |
| Reset last commit and discard changes | `git reset --hard HEAD~1` |
| Revert a commit (creates a new commit) | `git revert 123456` |
| Remove untracked files | `git clean -fd` |
| Preview untracked files to remove | `git clean -fdn` |

### Conflicts

| Description | Command |
| --- | --- |
| Keep our version of a file | `git restore --ours /path/to/file` |
| Keep their version of a file | `git restore --theirs /path/to/file` |
| Keep our version of all files | `git restore --ours .` |
| Keep their version of all files | `git restore --theirs .` |

## Workflows

### Rebase `main` on `my-branch`

1. Switch to and update `my-branch`:
   ```
   git switch my-branch
   git pull
   ```
2. Switch to `main` and rebase:
   ```
   git switch main
   git rebase my-branch
   ```
3. Push the changes:
   ```
   git push --force-with-lease
   ```

### Merge `my-branch` into `main`

1. Switch to and update `my-branch`:
   ```
   git switch my-branch
   git pull
   ```
2. Switch to `main` and merge:
   ```
   git switch main
   git merge my-branch
   ```
3. Push the changes:
   ```
   git push --force-with-lease
   ```

### Squash commits

Squash several commits into one before merging to the main branch.

1. Identify the hash of the last commit you don't want to squash:
   ```
   git log
   ```
2. Interactive rebase on that commit (e.g. `123456`):
   ```
   git rebase -i 123456
   ```
3. In the editor, replace `pick` with `f` on every line except the first. Save and edit the final commit message.

### Bisect

Find which commit introduced a bug using binary search.

1. Start bisect:
   ```
   git bisect start
   ```
2. Mark the current commit as bad:
   ```
   git bisect bad
   ```
3. Mark a known good commit:
   ```
   git bisect good 123456
   ```
4. Git will checkout a commit in between. Test it and mark it:
   ```
   git bisect good   # if the bug is not present
   git bisect bad    # if the bug is present
   ```
5. Repeat step 4 until git identifies the culprit. Then reset:
   ```
   git bisect reset
   ```

### Cherry-pick

Apply a specific commit from another branch.

1. Switch to the target branch:
   ```
   git switch branch-a
   ```
2. Find the commit hash to cherry-pick:
   ```
   git log branch-b
   ```
3. Cherry-pick the commit (e.g. `123456`):
   ```
   git cherry-pick -x 123456
   ```
4. Verify and push:
   ```
   git log
   git push
   ```
