Dependencies
---

```
sudo apt install python3-numpy python3-scipy python3-sklearn
```

Code Style
---

Please do not commit code with significant PEP8 violations. It's best to check
this with a pre-commit hook:

```
#!/bin/sh

if git rev-parse --verify HEAD >/dev/null 2>&1
then
	against=HEAD
else
	# Initial commit: diff against an empty tree object
	against=4b825dc642cb6eb9a060e54bf8d69288fbee4904
fi

# Redirect output to stderr.
exec 1>&2

git diff --cached $against | flake8 --diff
```
