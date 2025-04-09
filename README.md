# OpenSourceExample
Example configuration for repositories that will become open-source/source-available.

## Using the `trufflehog` Pre-Commit Hook
This repository includes a pre-commit hook that uses the `trufflehog` tool to scan your code for secrets before each commit. This helps prevent secrets, such as API keys and passwords, from being accidentally committed to the repository.

### Prerequisites
Install `pre-commit` by running:
```bash
pip3 install pre-commit
```
Before you can use the `trufflehog` pre-commit hook, you need to have the `trufflehog` tool installed. You can install it using the following command:
```bash
brew install trufflehog
```
Once you have both tools installed, you can run `pre-commit install` to install the pre-commit hooks in your repository:

### Using the Pre-Commit Hook
Once you have the `trufflehog` tool installed and have added the patterns you want to search for (OAI keys added by default), you can use the pre-commit hook to automatically scan your code before each commit. To use the pre-commit hook, simply run the `git commit` command as you normally would. 

The `trufflehog` tool will automatically scan your code for secrets and reject the commit if any are found. If any secrets are found, you will be prompted to remove them before trying.

