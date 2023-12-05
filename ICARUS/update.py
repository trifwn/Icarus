import git

from ICARUS import APPHOME

repo = git.Repo(APPHOME)
repo.remotes.origin.pull()

# Get the latest commit hash
current = repo.head.commit

# Get the latest commit hash from remote
repo.remotes.origin.fetch()

# Compare the two hashes
print(f"Repo is at {current}")
print(f"Remote is at {repo.remotes.origin.url}")
if current == repo.remotes.origin.refs.main.commit:
    # up to date
    print("Up to date")
else:
    # not up to date
    print("Not up to date")
    repo.remotes.origin.pull()
    print("Updated local repository to match remote")
