# Git Cheatsheet

The minimum you need to be productive.

## Setup (once)

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global pull.rebase false   # default merge on pull
```

## Starting a repository

```bash
git init                  # create a new local repo
git clone <url>           # copy a remote repo locally
```

## The everyday loop

```bash
git status                # what's changed?
git diff                  # what's the actual change in unstaged files?
git diff --staged         # what's staged for commit?

git add file.py           # stage one file
git add .                 # stage all changes (be careful — review with status first)

git commit -m "Add feature X"

git log --oneline --graph --decorate --all   # pretty history
```

## Branches

```bash
git branch                  # list branches
git switch -c feature/x     # create + switch to new branch
git switch main             # jump to main
git merge feature/x         # merge feature/x into current branch
git branch -d feature/x     # delete after merge
```

> Older syntax: `git checkout -b name` ≡ `git switch -c name`.

## Working with remotes

```bash
git remote -v
git remote add origin git@github.com:you/repo.git

git push -u origin main     # push and set upstream (only first time)
git push                    # subsequent pushes
git pull                    # fetch + merge

git fetch                   # download remote refs without merging
```

## Undoing things

```bash
git restore file.py              # discard unstaged changes to file.py
git restore --staged file.py     # un-stage (keep edits)
git commit --amend               # edit the last commit (only if not pushed)
git reset --soft HEAD~1          # undo last commit, keep changes staged
git reset --hard HEAD~1          # ⚠️ throw away last commit AND changes
git revert <sha>                 # create a new commit that undoes <sha> (safe)
```

## Inspecting

```bash
git log -p file.py        # see history of a file
git blame file.py         # who last changed each line
git show <sha>            # full diff of a commit
```

## A typical PR workflow on GitHub

```bash
git switch -c feature/add-thing
# edit files...
git add . && git commit -m "Add thing"
git push -u origin feature/add-thing
# Open the URL git prints, click "Compare & pull request"
```

Or use the [GitHub CLI](https://cli.github.com/):

```bash
gh pr create --fill
gh pr view --web
```

## Useful aliases (paste into your shell config)

```bash
alias gs='git status'
alias gd='git diff'
alias gco='git switch'
alias gl='git log --oneline --graph --decorate --all -20'
```

## Things to know

- **A commit is forever.** Until you push, it's local — but be careful with `reset --hard`.
- **Pull before you push** if working with others.
- **Small, focused commits** are far easier to review and revert.
- **Never commit secrets.** Add them to `.gitignore` and use a `.env.example` template.
