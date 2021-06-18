# Git

## Commands


### Differences


| Description | Command |
| --- | --- |
| Display diff | `git diff` |
| Display diff with HEAD | `git diff HEAD` |
| Display staged diff | `git diff --cached` |


### Staged changes

| Description | Command |
| --- | --- |
| Staged changes of one file | `git add /path/to/file` |
| Staged all changes (do not use) | `git add .` |
| Staged changes interactively (preferred) | `git add -p` | 
| Discard local changes of one file | `git checkout HEAD /path/to/file` |


### Commit


| Description | Command |
| --- | --- |
| Commit (vi interface) | `git commit` |
| Quick commit | `git commit -m "commit message"` |
| Amend a commit | `git commit --amend` <br> `git push --force-with-lease` |
| Amend a commit | `git ammend` |
 


### Branches

| Description | Command |
| --- | --- |
| List local branches | `git branch` |
| List all branches | `git branch -a` |
| Checkout a branch | `git checkout my-branch` |
| Create branch | `git checkout -b my-branch` |
| Push local branch | `git push -u origin my-branch` |
| Remove local branch | `git branch -d my-branch` <br> `git branch -D my-branch` |
| Remove distance branch | `git push origin --delete my-branch` |
| Update the list of branches | `git remote update origin --prune` |


### Useful

| Description | Command |
| --- | --- |
| Rebase `my-branch` on `main` (several commands) | `git checkout my-branch`<br>`git pull`<br>`git checkout main`<br>`git rebase my-branch`<br>`git push --force-with-lease` |
| Merge `my-branch` on `main` (several commands) | `git checkout my-branch`<br>`git pull`<br>`git checkout main`<br>`git merge my-branch`<br>`git push --force-with-lease` |
| Reset last commit and keep changes  | `git reset --soft HEAD~1` |
| Reset last commit and discard changes  | `git reset --hard HEAD~1` |


### Squash commits

You're working on a branch and you have several commits. Before merging to the main branch, some people like to "squash" commits into one single commit.

Identify the hash of last commit you don't want to squash. Let's say it's 123456:

```
$ git log
```

Rebase on this commit:

```
$git rebase -i 123456
```

This will bring you to an interface that will allow you to edit the commit. Except for the first line, replace by "f" for squashing the commits. Then you can edit the name of the single commit.


### Cherry-pick

Checkout the branch you want to modify:

```
$ git checkout branch-a
```

List all the commits of the branch you want to cherry pick from to identify the correct commit. Let's say your commit is 123456.

```
$ git log branch-b
```

Cherry pick the commit 123456:

```
$ git cherry-pick -x 123456
```

Verify you have correctly cherry-picked:

```
$ git log
```

Push the modifications:

```
$ git push
```
