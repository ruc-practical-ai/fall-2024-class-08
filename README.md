# Loaded Lecture Template

Template for lecture notes with reveal.js presentations, latex dependencies, poetry, and VS Code extensions for Python.

## Installation and Usage

### Installation and Use via Github Codespaces (Recommended)

To use this repository via codespaces simply click on the `code` &rarr; `codespaces` &rarr; `create codespace on main` buttons.

Once the codespace is open in the browser, click the three bars in the top left corner and select `Open in VS Code Desktop`.

Widgets might work better when using VS Code Desktop vs. in the browser.

Note the codespace might take a long time to build. This is usually due to TexLive dependencies. Use `Cmd` / `Ctrl` + `Shift` + `P` &rarr; `Codespaces: View Creation Logs` to check status.

If required, use `Cmd` / `Ctrl` + `Shift` + `P` &rarr; `Codespaces: Rebuild Container` to rebuild the container. Do not use `gh codespace rebuild`. This takes a long time since it re-downloads the entire image.

### Display of Presentations via Github Codespaces (Recommended)

Navigate to the `presentations` folder.

```bash
cd presentations
```

Start an http-server.

```bash
bash ../scripts/start_server.sh
```

### Dependencies for Local Installation

#### Reveal.js

This project is built on [reveal.js](https://revealjs.com/). All reveal.js dependencies are included in the repository. The repository itself is a modified [basic setup](https://revealjs.com/installation/#basic-setup) of reveal.js.

#### Poetry

This project is built on Python 3.12. Poetry is required for installation. To install Poetry, view the instructions [here](https://python-poetry.org/docs/).

In codespaces, Poetry installation is handled in the development container. The user does not need to install Poetry if working in codespaces.

#### TexLive

This project also requires TexLive to render math fonts. Texlive can be installed via the following commands.

```bash
sudo apt-get -y update
sudo apt-get -y install texlive
sudo apt-get -y install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```

In codespaces, TexLive installation is also handled in the development container. The user does not need to install these packages if working in codespaces.

### Local Installation

To install locally, first install the required dependencies (Poetry and TexLive), then clone the repository and navigate to its directory.

```bash
git clone https://github.com/ruc-practical-ai/example-reveal-js-presentation.git
cd example-reveal-js-presentation
```

#### Viewing HTML Pages Directly in a Browser from Local Installation

To view HTML pages directly in a browser, simply navigate to the pages of interest and open them with a preferred web browser.

#### Installing Python Dependencies Locally

To install locally, first install the required dependencies (Poetry and TexLive), then clone the repository and navigate to its directory.

```bash
git clone https://github.com/ruc-practical-ai/loaded-lecture-template.git
cd loaded-lecture-template
```

Configure Poetry to install its virtual environment inside the repository directory.

```bash
poetry config virtualenvs.in-project true
```

Install the repository's Python dependencies.

```bash
poetry install --no-root
```

Check where Poetry built the virtual environment with the following command.

```bash
poetry env info --path
```

Open the command pallette with `Ctrl` + `Shift` + `P` and type `Python: Select Interpreter`.

Now specify that VSCode should use the that interpreter (the one in `./.venv/Scripts/python.exe`). Once you specify this, Jupyter notebooks should show the project's interpreter as an option when you click the `kernel` icon or the small icon showing the current version of python (e.g., `Python 3.12.1`) and then click `Select Another Kernel`, and finally click `Python Environments...`.

## License

This repository is provided with an MIT license. See the `LICENSE` file.

## Contributing

Please email Mauro Sanchirico at ms3978@camden.rutgers.edu (academic) or sanchirico.mauro@gmail.com (personal) with questions, comments, bug reports, or suggestions for improvement.

