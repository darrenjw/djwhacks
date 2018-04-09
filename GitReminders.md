# Git reminders

## Config and setup

* `git config --global user.name "Firstname Lastname"`
* `git config --global user.email foo@bar.com`

## Cloning repos

* `git clone git@github.com:darrenjw/djwhacks.git`   (checkout into current dir)

## Common commands

* `git pull`             (update)
* `git add file.c`       (add a file to the repo)
* `git commit -a`        (commit locally)
* `git push`             (push local commits back to github)
* `git status`           (useful info)
* `git log`              (commit log)
* `git tag`              (list of tagged commits)
* `git tag v0.3`         (tag a version)
* `git push origin v0.3` (push tag to remote)

## Changing upstream

If you accidentally clone using "https" you will be asked for username and password. You can fix with:

`git remote -v`

to check current origin and then:

`git remote set-url origin git@github.com:username/repo.git`

to replace it with something appropriate.


## Keeping a fork in sync with an upstream:

First add add an upstream

```bash
git remote -v
git remote add upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git
git remote -v
```

Check it looks like it worked

Then:

```bash
git fetch upstream
git checkout master
git merge upstream/master
git push
```

## Branching and merging

Create a new branch and switch to it
```bash
git checkout -b my-branch
```

Shorthand for:
```bash
git branch my-branch
git checkout my-branch
```

Commit changes before switching branches.
Push new branch back to GitHub with:
```bash
git push origin my-branch
```
Can then do a pull request (on GitHub) for merging into master.

Switch back to `master` with:
```bash
git checkout master
```

Delete branch with:
```bash
git branch -d old-branch
```

Pull in changes to current branch from `master` with:
```bash
git merge master
```

Pull changes from a branch into master with:
```bash
git checkout master
git merge good-branch
```

## Undo a commit

If you commit a change you regret and you haven't yet pushed it, you can undo it with:
```bash
git reset --hard HEAD^
```
Note that you shouldn't do this if you've already pushed, as things will get out of sync.


## Web links

* [Pro Git](https://git-scm.com/book/en/v2/) (book)
  * [Basic branching and merging](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) (including conflict resolution)
* [Learn Git in 30 Minutes](http://tutorialzine.com/2016/06/learn-git-in-30-minutes/)
* [Git on the command line](http://dont-be-afraid-to-commit.readthedocs.io/en/latest/git/commandlinegit.html)  
* [Getting Git Right](https://www.atlassian.com/git/)
* [Lesser known Git commands](https://hackernoon.com/lesser-known-git-commands-151a1918a60)
* [Git from the inside out](https://codewords.recurse.com/issues/two/git-from-the-inside-out)

### eof


