# Terminal / Unix command reference

Running log of command-line commands learned while working on this
project. macOS, Linux, and (via WSL) Windows all share this same Unix
command family — this reference isn't Mac-specific, it transfers to any
Linux server (cloud GPU boxes, AWS EC2, etc.) used later in the project.

## Navigation

| Command | What it does |
|---|---|
| `pwd` | Print working directory — shows where you currently are |
| `ls` | List files/folders in the current directory |
| `ls <folder>` | List contents of a folder without moving into it |
| `cd <folder>` | Change directory — move into a folder |
| `cd ~` | Go to home directory |
| `cd ..` | Go up one directory level |

## Common mistakes (and why they happen)

- **`cd~` (no space) fails.** Most commands need a space between the
  command and its argument — without one, the shell reads it as a single
  unrecognized word.
- **`cd <file>.tar.gz` fails.** `cd` only works on directories, never on
  files.
- **`LS` / `CD` (capitals) fail.** Unix commands are case-sensitive;
  only the exact lowercase command works.
- **Running `ls Downloads` while already inside `Downloads`** looks for
  a nonexistent nested `Downloads/Downloads` folder. If a path-based
  command behaves unexpectedly, run `pwd` first to confirm where you
  actually are.

## Git (assumed known, quick reference)

| Command | What it does |
|---|---|
| `git init` | Turn the current folder into a git repository |
| `git add -A` | Stage all changed files for commit |
| `git commit -m "message"` | Save a snapshot of staged changes |
| `git remote add origin <url>` | Link the local repo to a GitHub repo |
| `git push -u origin main` | Push commits to GitHub for the first time |
| `git push` | Push new commits after the first push |
| `git status` | Show what's changed / staged / untracked |
| `git log --oneline` | Show commit history, one line per commit |

## To add as the project grows

- Process management (`ps`, `kill`) — useful once multiple worker
  processes are running at once (Phase 3).
- `curl` in more depth — testing APIs from the command line.
- `chmod` / permissions — likely needed once deploying to a remote
  machine (cloud GPU, EC2).
