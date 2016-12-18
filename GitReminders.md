# Git reminders

## Config and setup

* `git config --global user.name "Firstname Lastname"`
* `git config --global user.email foo@bar.com`

## Cloning repos

* `git clone git@github.com:darrenjw/djwhacks.git`   (checkout into current dir)

## Common commands

* `git pull`       (update)
* `git add file.c` (add a file to the repo)
* `git commit -a`  (commit locally)
* `git push`       (push local commits back to github)
* `git log`        (commit log)
* `git tag`        (list of tagged commits)
* `git tag v0.3`   (tag a version)

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

## Web links

* [Learn Git in 30 Minutes](http://tutorialzine.com/2016/06/learn-git-in-30-minutes/)
* [Getting Git Right](https://www.atlassian.com/git/)
* [Lesser known Git commands](https://hackernoon.com/lesser-known-git-commands-151a1918a60)
* [Git from the inside out](https://codewords.recurse.com/issues/two/git-from-the-inside-out)

### eof


