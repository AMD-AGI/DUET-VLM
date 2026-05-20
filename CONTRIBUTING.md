# Contributing

Thank you for your interest in contributing! This document describes how to
propose changes, the review process, and the coding standards we follow.

This guide aligns with the
[AMD-AGI GitHub Repository Standards](https://github.com/AMD-AGI/)
and the [OpenSSF Best Practices Badge](https://www.bestpractices.dev/) (Passing
level).

## Code of Conduct

Be respectful, collaborative, and inclusive. Harassment or abusive behavior
will not be tolerated.

## Ways to Contribute

- **Report a bug** — open an issue with reproduction steps, the expected vs.
  actual behavior, your environment (OS, GPU, Python, ROCm/CUDA version), and
  any relevant logs.
- **Request a feature** — open an issue describing the use case, proposed
  behavior, and why an existing workaround is insufficient.
- **Submit a pull request** — see below.

## Development Workflow

1. **Fork the repository** (or, if you have write access, create a branch from
   the default branch).
2. **Create a feature branch:**
   ```bash
   git checkout -b feat/short-description
   ```
3. **Set up the development environment** per the project README.
4. **Make your changes**, keeping commits small and focused.
5. **Run tests and linters locally** before pushing:
   ```bash
   # adjust to project-specific commands
   pytest
   ruff check .
   ```
6. **Open a pull request** against the default branch. Link any related issues
   (`Fixes #123`).

## Pull Request Guidelines

- Keep PRs scoped — one logical change per PR.
- Update or add tests covering your change.
- Update documentation (README, docstrings, examples) when behavior changes.
- Ensure CI passes before requesting review.
- A review from a [CODEOWNER](.github/CODEOWNERS) is required before merge.
- Squash commits on merge unless preserving history is intentional.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) where
practical:

```
<type>(<scope>): <short summary>

<body — optional, what and why>
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.

## Coding Standards

- **Python** — follow PEP 8; format with `ruff format` or `black`; type hints
  encouraged on public APIs.
- **Docstrings** — Google or NumPy style for public functions and classes.
- **Tests** — `pytest` for unit tests; place them under `tests/`.
- **Imports** — keep them sorted (`ruff check --select I` or `isort`).

## Reporting Security Issues

Please do **not** report security vulnerabilities through public GitHub issues.
See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

## Licensing

By contributing, you agree that your contributions will be licensed under the
project's [LICENSE](LICENSE). If your contribution includes third-party code,
ensure its license is compatible and note it in the PR description.

## Questions

For internal AMD-AGI contributors, reach out to the team listed in
[CODEOWNERS](.github/CODEOWNERS). External contributors should open a
discussion or issue.
