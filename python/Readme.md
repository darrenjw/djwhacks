# Readme

## Python code

My python code directory

### rst

reStructuredText markup: https://devguide.python.org/documentation/markup/

### virtualenv

Reminders for `virtualenv`: https://pythonbasics.org/virtualenv/ 

```bash
virtualenv -p python3 myEnv

./myEnv/bin/activate
source ./myEnv/bin/activate

deactivate
```

### uv

Docs for `uv`: https://docs.astral.sh/uv/

```bash
# virtual environments
uv venv # creates .venv
uv venv myEnv
uv venv --python 3.12 myEnv
source ./myEnv/bin/activate
uv pip install black
deactivate

# create a new project
uv init myproject
cd myproject
uv add black # adds to project dependencies (and the .venv)

# update (if installed with the standalone installer)
uv self update
uv --version

```
