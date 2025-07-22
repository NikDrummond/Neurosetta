# Neurosetta

### Installation

This is currently very hacky. 

You need `conda`! 

**For Linux - and I *think* Mac:**

1) Clone the repository from github onto your local machine.
2) in the command line, `cd` to the root of the Neurosetta repo.
3) create a virtual environment using the `environment.yml` file provided like so:

```bash
conda env create -f environment.yml
```

This will create an environment called `neurosetta` which has the minimal requirements. Activate this environment using:

```bash
conda activate neurosetta
```

Once you have activated the `neurosetta` conda environment, and still from the root directory of Neurosetta, you can install it locally with:

```bash
python3 -m pip install .
```

**For Windows**

`graph-tool` - the graph theory toolbox neurosetta is built on, is  bit tricky to installl in windows. I only have this working using Visua Studio Code as you need the WSL extension.

First, install Ubuntu using wsl:

``` bash
wsk --install 
```

You then need to restart, and open ubuntu in your start bar. See [here](https://learn.microsoft.com/en-us/windows/wsl/install) if you have any issues.

Once in the ubuntu temrinal, you need to install miniconda:

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the installer
bash Miniconda3-latest-Linux-x86_64.sh

# Follow the prompts and restart your shell
```

then, you want to navigate to where your have placed the Neurosetta repository in windows. to get there from WSL your path should be something like:

```bash
cd /mnt/c/Users/<yourusername>/<the path in windows>
```

You can then repeat the steps described above as if you were installing using Ubuntu.

While here, and the neurosetta environment is activated, quickly use:

``` bash
which python
```

and make a note of the output, something like `/home/yourusername/miniconda3/envs/neurosetta/bin/python`. we will need this in a second!

ok, you have created your environment and have neurosetta installed, but it is WSL.

To *find* the environemnt, open VSCode and make sure you have the WSL extension (should be easy to find in the extensions marketplace...)

open VSCode and do the following:

1) Press `Ctrl + Shift + P`

2) Run: `Python: Select Interpreter`

3) Wait a few seconds — VS Code should automatically list WSL Conda environments

If neurosetta doesn't appear:

    Scroll down and click “Enter interpreter path”

    Then click “Find...” (don’t paste yet!) — this will open a WSL file browser

    Now navigate to:

        the path we noted before!!!

        Select it

This should do the trick. Or just don't use windows. 
