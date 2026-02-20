# auditory-pRF-subcortical

- contact: e.celikay@bcbl.eu / i.ekimcelikay@gmail.com
-----
#### Quick access:
- [Commits-main](https://github.com/iekimcelikay/auditory-pRF-subcortical/commits/main/)
-----

Last edit to the repo: 20/02/2026

https://github.com/iekimcelikay/cochlea.git install cochlea py3-zilany2014 from here. 

---
## IMPORTANT:
- If a file has a date in its name e.g. `280126` it means (in most cases) it is my working script. This does not entail that I won't make changes in the other scripts, but the others are more likely to be stable. The ones with the dates are the ones I'm currently working on to do some new stuff. 
- You need to run the script in project root for the imports to work. In the future, this may be converted into a proper package. For now, it's modules. 
- Use the cochlea package in this repo, its' Zilany2014 model codes are updated for Python 3.
- To make thorns work in Python3, these changes might be needed:
-> replace `import imp` with `import importlib as imp`
-> replace `from collections` with `collections.abc

## Changelog

### 2026-01-30: Repository name change, creating of changelog_archive, updated README

**Repository name changed**
- old: `cochlear-models`
- new: `auditory-pRF-subcortical`
- If you have a local copy, run the command `git remote set-url origin  https://github.com/iekimcelikay/auditory-pRF-subcortical` inside the directory. (source: [Github Docs: Creating and managing repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository)

**Created a changelog_archive**:
- To keep the readme file readable, a creation of changelog_archive.md that will keep the old changelog updates (current-1).

**Notes on thorns package and python 3 added to readme**:
> - To make thorns work in Python3, these changes might be needed:
-> replace `import imp` with `import importlib as imp`
-> replace `from collections` with `collections.abc

________________________________________________________________________________________________
File created on: 26/01/26

