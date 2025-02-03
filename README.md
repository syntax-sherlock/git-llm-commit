# git-llm-commit

[![codecov](https://codecov.io/github/syntax-sherlock/git-llm-commit/graph/badge.svg?token=YZECGT1JIF)](https://codecov.io/github/syntax-sherlock/git-llm-commit)

Generate Conventional Commit messages from your staged changes using an LLM (GPT-4).

## Description

`git-llm-commit` is a command-line tool that analyzes your staged git changes and generates a commit message following the [Conventional Commits](https://www.conventionalcommits.org/) specification using OpenAI's GPT-4 model.

## Installation

1. Ensure you have Python 3.x installed
2. Install the package:
   ```bash
   pip install git-llm-commit
   ```

## Setup

1. Get an OpenAI API key from [OpenAI's platform](https://platform.openai.com/)
2. Set your API key in your environment:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```
   Or create a `.env` file in your project root with:
   ```
   OPENAI_API_KEY=your-api-key
   ```

## Usage

1. Stage your changes as usual:

   ```bash
   git add .
   ```

2. Instead of `git commit`, run:

   ```bash
   git llm-commit
   ```

3. Review the generated commit message and:
   - Press `y` to accept and commit
   - Press `n` to abort
   - Press `e` to edit the message before committing

## Features

- Generates commit messages following Conventional Commits format
- Uses GPT-4 to analyze diffs and create meaningful commit messages
- Supports interactive editing of generated messages
- Integrates with your default git editor
- Respects conventional commit types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
- Detects potentially risky files (like .env, credentials, secrets) and prompts for confirmation before committing

## Requirements

- Python 3.x
- Git
- OpenAI API key

## License

[MIT License](LICENSE)
