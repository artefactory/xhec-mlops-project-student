<div align="center">

# xhec-mlops-project-student

[![CI status](https://github.com/artefactory/xhec-mlops-project-student/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory/xhec-mlops-project-student/actions/workflows/ci.yaml?query=branch%3Amaster)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)]()

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>


# Abalone Age Prediction Industrialization

This repository is dedicated to industrializing the [Abalone age prediction Kaggle contest](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset). The project focuses on streamlining the machine learning pipeline, implementing best practices, and ensuring seamless deployment through the use of pre-commit hooks, Docker, and requirements files.

<details>
<summary>Details on the Abalone Dataset</summary>

The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age.

**Goal**: predict the age of abalone (column "Rings") from physical measurements ("Shell weight", "Diameter", etc...)

You can download the dataset on the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)

</details>

## Table of Contents
still to adapt
- [xhec-mlops-project-student](#xhec-mlops-project-student)
- [Abalone Age Prediction Industrialization](#abalone-age-prediction-industrialization)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
    - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)


## Project Overview

The Abalone age prediction project aims to predict the age of abalones based on various physical features. The dataset, available on Kaggle, serves as the foundation for this machine learning endeavor.

### Features

- **Pre-commit Hooks:** The project integrates pre-commit hooks to ensure consistent code style, formatting, and best practices before each commit. This helps maintain a clean and standardized codebase.

- **Docker Integration:** The repository includes a Dockerfile for easy containerization of the project. Docker ensures that the environment is reproducible across different systems, simplifying deployment and collaboration.

- **Requirements File:** The requirements file (`requirements.txt`) lists all the dependencies necessary to run the project. This includes libraries such as NumPy, Pandas, scikit-learn, and any other packages essential for data processing and machine learning tasks.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Docker installed on your system.
- Python 3.x and pip for managing Python packages.

### Installation
1. Clone the repository to your local machine:
```bash
git clone https://github.com/ConstanceGlly/xhec-mlops-project-student.git
cd abalone-age-prediction
```
2. Build the Docker image:
```bash
docker build -t abalone-prediction .
```
3. Run the Docker container:
```bash
docker run -p 5000:5000 abalone-prediction
```

The project should now be running locally, and you can access it at `http://localhost:5000`.

## Contributing

If you would like to contribute to the project, please follow these guidelines:

1. Fork the repository on GitHub.
2. Create a new branch with a descriptive name: `git checkout -b feature-name`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request detailing the changes you made.

## License

This project is licensed under the MIT License - see the [LICENSE](MIT-LICENSE.txt) file for details.

## Acknowledgments

- Kaggle for providing the Abalone dataset.
