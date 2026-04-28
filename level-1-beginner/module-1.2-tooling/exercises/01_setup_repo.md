# Exercise 1 — Set up a Git + GitHub workflow

**Goal:** practice the everyday loop of branching, committing, pushing, and opening a pull request.

## Steps

### 1. Create a repo on GitHub

- Go to https://github.com/new
- Name: `python-tooling-practice`
- Visibility: private or public, your call
- Initialize **without** a README (we'll push our own)
- Click **Create repository**

GitHub will show you the commands to run. We'll do them ourselves below.

### 2. Initialize locally

```bash
mkdir python-tooling-practice
cd python-tooling-practice

git init
git branch -M main
```

### 3. Add a file and make the first commit

Create `hello.py`:

```python
def hello(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(hello("world"))
```

Then:

```bash
git add hello.py
git commit -m "Initial commit: hello function"
```

### 4. Connect to GitHub and push

```bash
git remote add origin git@github.com:YOUR-USERNAME/python-tooling-practice.git
git push -u origin main
```

> Use HTTPS (`https://github.com/...`) instead of SSH if you haven't set up SSH keys.

### 5. Make a feature branch

```bash
git switch -c feature/add-greeting-language
```

Edit `hello.py`:

```python
def hello(name: str, lang: str = "en") -> str:
    greetings = {"en": "Hello", "fr": "Bonjour", "es": "Hola"}
    return f"{greetings[lang]}, {name}!"
```

Then:

```bash
git add hello.py
git commit -m "Support multiple languages in hello()"
git push -u origin feature/add-greeting-language
```

### 6. Open a pull request

- Go to your repo on GitHub.
- You should see a banner: **"Compare & pull request"**.
- Title: `Support multiple languages in hello()`
- Description: a sentence or two about what changed and why.
- Click **Create pull request**.

### 7. Merge it

- On the PR page, click **Merge pull request** → **Confirm merge**.
- Locally:

```bash
git switch main
git pull
git branch -d feature/add-greeting-language
```

You're back on `main`, in sync with GitHub, and the feature branch is cleaned up.

## Bonus

- Try `gh pr create --fill` from the terminal using the [GitHub CLI](https://cli.github.com/).
- Set up a `.gitignore` (copy [`../.gitignore`](../.gitignore)).
- Add a `README.md` and push it as a second PR.

## Checkpoint

You can: create a repo, branch, commit, push, open a PR, merge it, and sync your local main — without looking up the commands.
