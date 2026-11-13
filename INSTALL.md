# Running VankeDoodle on your own computer

Written for someone who has never used a terminal. Follow it in order; every
command is one line that you copy, paste, and press Enter. Windows first, with
Mac notes at the end.

You do this whole thing **once**. After that, starting the app is two commands
(["Every time after that"](#every-time-after-that) below).

---

## Step 1 - Install Python

Python is the language the engine is written in.

**First check whether you already have it.** Press the Windows key, type
`powershell`, press Enter, and in the blue window type:

```powershell
python --version
```

If that prints something like `Python 3.14.5`, you already have Python - skip
straight to Step 2. If it prints an error, carry on.

Windows installs Python through the **Python install manager** now. It is a small
app that fetches and keeps Python versions up to date for you. Either route works:

- **From the Microsoft Store (easiest):** open the Store, search for
  **Python Install Manager**, click **Install**.
- **From the web:** go to <https://www.python.org/downloads/> and click the big
  button, which says **Download Python Install Manager**. Open the downloaded
  file and choose **Install**.

Then close PowerShell, open it again, and type:

```powershell
python --version
```

The first time you run this, the manager will notice that no Python version is
installed yet and offer to install the latest one. Say yes. If it asks permission
to add a folder to your PATH, say yes to that too. When it finishes, run
`python --version` once more and you should see a version number.

> If you are following along with an older guide or video, you may see an
> instruction to tick a box labelled **"Add python.exe to PATH"** during install.
> That checkbox belonged to the previous installer and no longer exists - the
> install manager handles it. There is nothing you need to tick.

> If the manager does not offer to install a version, ask it directly:
> ```powershell
> py install default
> ```

## Step 2 - Put the project folder somewhere you can find

Get the project onto your computer. If it is published on GitHub, open the
repository page, click the green **Code** button, choose **Download ZIP**, then
right-click the ZIP and choose **Extract All**. Otherwise just use the folder of
files you already have.

Put it somewhere simple, for example:

```
C:\Users\YourName\Documents\vankedoodle
```

Open that folder. You should see `app.py`, `requirements.txt`, `pyproject.toml`
and a folder called `vankedoodle`. If instead you see *another* folder with the
same name, go into it - that inner one is the project.

## Step 3 - Open a terminal in that folder

In File Explorer, with the project folder open, click the address bar at the top,
type `powershell`, and press Enter. A blue window opens, already pointed at the
right folder.

To check it worked, type this and press Enter:

```powershell
python --version
```

You should see something like `Python 3.13.1`. If you get an error instead, see
["If something goes wrong"](#if-something-goes-wrong).

## Step 4 - Create a private workspace for the project

This keeps VankeDoodle's parts separate from anything else on your computer, so
nothing can break anything else.

```powershell
python -m venv .venv
```

Nothing visible happens; it takes a few seconds. Then switch into it:

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. **That prefix is how you know the
workspace is active.** If you ever see commands fail later, check it is there.

> If this step gives a red error about "running scripts is disabled", run this
> line once and then try activating again:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```
> It only affects this one window and is undone when you close it.

## Step 5 - Install VankeDoodle and the things it needs

Two commands. The first takes a minute or two and prints a lot of text — that is
normal.

```powershell
pip install -r requirements.txt
pip install -e .
```

The second one installs the engine itself so the app can find it. Do not skip it.

## Step 6 - Start the app

```powershell
streamlit run app.py
```

Your web browser opens automatically at `http://localhost:8501` showing the app.
If it does not open by itself, copy that address into your browser.

The first time, Streamlit may ask for an email address. Just press Enter to skip it.

**The app is running on your own computer.** Nothing you type is sent anywhere.
The black window must stay open while you use it — minimise it, do not close it.

## Step 7 - Stop the app

Click on the black window and press `Ctrl` and `C` together. The app stops.
Then close the window.

---

## Every time after that

You do not repeat the setup. To start the app again:

1. Open the project folder, click the address bar, type `powershell`, press Enter.
2. Run these two lines:

```powershell
.venv\Scripts\Activate.ps1
streamlit run app.py
```

That is it. Consider saving those two lines somewhere you can copy them from.

---

## What you will see

The page has the patient on the left (weight, age, height, creatinine), the doses
given and levels drawn in the middle, and the results underneath. It starts
pre-filled with an example patient so you can see it working immediately — type
over the values with your own.

Results appear as soon as you change something; there is no "calculate" button.

- **Where the patient is now** - the AUC24 on the regimen they are currently on,
  with its uncertainty and the chance it is in target.
- **Recommended next regimen** - the dose and interval that put the AUC on target.
- **Concentration over time** - grey is what the population model expected, blue
  is after fitting the measured levels (red dots), green dashed is what the
  recommended regimen would do. Hover over any line to read values.
- **Two-level calculation** - the Sawchuk-Zaske result, when two levels in one
  dosing interval allow it. When they do not, it tells you why.
- **Save this case** - appends a one-line summary to `cases.jsonl` in the project
  folder, for the study database.

Tick **Compare all models** in the sidebar to run all six population models at
once and see how much they disagree.

---

## If something goes wrong

**`python is not recognized as the name of a cmdlet`**
The Python install manager is not installed, or this window was opened before it
was. Close PowerShell, open it again, and try once more. If it still fails, go
back to Step 1 and install the Python install manager.

**`running scripts is disabled on this system`**
See the box in Step 4.

**`ModuleNotFoundError: No module named 'vankedoodle'`**
Either the `(.venv)` prefix is missing from your prompt — run
`.venv\Scripts\Activate.ps1` — or you skipped `pip install -e .` in Step 5.

**`ModuleNotFoundError: No module named 'streamlit'`**
You skipped `pip install -r requirements.txt`, or you are not in the `(.venv)`.

**`Port 8501 is already in use`**
The app is already running in another window. Either use that one, or start this
one on a different port:
```powershell
streamlit run app.py --server.port 8502
```

**The browser opens but the page is blank or spins forever**
Wait ten seconds and refresh. If it persists, stop with `Ctrl+C` and start again.

**`Could not find an appropriate Python runtime` or a prompt you dismissed**
The install manager is there but no Python version is. Run:
```powershell
py install default
```

**Something else**
Run the test suite — it checks the engine and the app without you having to click
anything:
```powershell
pip install pytest
pytest -q
```
`96 passed` means everything is installed correctly and the problem is elsewhere.

---

## On a Mac

There is no install manager on macOS - download the normal installer from
<https://www.python.org/downloads/> and run it. Then open **Terminal**,
type `cd ` (with the space), drag the project folder onto the Terminal window, and
press Enter. Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

Afterwards, starting it again is `source .venv/bin/activate` then
`streamlit run app.py`.

---

## Using it without the app

Everything the page shows is available from the command line, which is handy for
checking a case quickly or scripting the study analysis:

```powershell
python -m vankedoodle start --weight 23 --age 7 --height 122 --scr 0.33
python -m vankedoodle evaluate example_case.json --record cases.jsonl
```

See `README.md` for the Python interface and `API.md` for the JSON contract.
