# Contributing to theblas

Thanks for contributing to theblas.

This guide explains how to set up your environment, make changes, and open high-quality pull requests.

## Project Basics

- Language: C++17
- Build system: CMake (minimum 3.15)
- Tests: CTest with tests in `tests/`

## Development Setup

### Option 1: Local environment

Requirements:

- CMake 3.15+
- A C++ compiler (GCC, Clang, or MSVC)

Build and test:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

### Option 2: GitHub Codespaces / Dev Container

This repository includes a ready-to-use dev container in `.devcontainer/`.

After startup, run:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## What to Contribute

Good contributions include:

- Bug fixes
- BLAS implementation improvements and extensions
- Cross-platform portability improvements
- Tests and documentation improvements

## Coding Guidelines

- Keep changes minimal and focused.
- Preserve public API stability unless an API change is intentional and documented.
- Keep code portable across GCC, Clang, and MSVC.
- Prefer clear and maintainable implementations over clever but hard-to-read code.
- Avoid unrelated refactors in the same pull request.

## Testing Expectations

When behavior changes, add or update tests in `tests/`.

Before opening a pull request, run:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

If you add new functionality, include tests for:

- Expected behavior
- Edge cases
- Any regressions fixed by your change

## Documentation Expectations

Update documentation when code changes affect behavior or usage.

Typical updates include:

- `README.md` for new features or changed usage
- API comments in public headers when public behavior changes

## Pull Request Guidelines

Please keep pull requests easy to review.

Checklist:

1. Build succeeds locally.
2. Tests pass locally.
3. Documentation is updated if needed.
4. Changes are scoped to one clear purpose.
5. Pull request description explains what changed and why.

Suggested PR description format:

- Summary
- Motivation
- Testing
- Notes on compatibility or API impact

## Maintainer Releases

Releases are triggered by pushing a version tag, not by merging a version bump.

1. On a release preparation branch, bump `project(theblas VERSION X.Y.Z ...)` in
   `CMakeLists.txt` and move the release notes from `[Unreleased]` to a dated
   `[X.Y.Z]` section in `CHANGELOG.md`, updating the comparison links.
   The optional local helper does both from the repository root:

   ```bash
   python3 scripts/prepare-release.py X.Y.Z
   python3 scripts/extract-changelog.py X.Y.Z
   ```

   Run preparation only once for a new version and review the diff. CI does not
   modify the CMake version or changelog.
2. Open a PR with these changes, validate it, and merge it into `master`.
3. Fetch `master` and tag the exact merged release commit (replace `X.Y.Z` and
   `<merged-release-commit>`):

   ```bash
   git fetch origin master
   git tag -a vX.Y.Z <merged-release-commit> -m "theblas vX.Y.Z"
   git push origin vX.Y.Z
   ```

4. The release workflow checks that the tag matches the CMake project version and
   that its commit is reachable from `origin/master`. It publishes a GitHub Release
   using that commit's versioned changelog section.
5. Review and merge the separate `release/vcpkg-port-vX.Y.Z` PR against `master`.
   It updates the archive SHA512, port version, registry tree hash, and baseline.

If the port update fails, rerun failed jobs in GitHub Actions. Rerunning the whole
workflow also works: an existing release skips only release creation. Existing port
PRs (including closed ones) are left untouched. If the branch was pushed but PR
creation failed, a rerun opens the PR without rewriting the branch.
An older release's update is skipped if a newer port version is already on `master`.

### Required Repository Settings

- Enable **Settings → Actions → General → Workflow permissions → Allow GitHub
  Actions to create and approve pull requests**. Organization policy may also need
  to permit this.
- A tag ruleset protecting `v*` is recommended: restrict tag creation to release
  maintainers and prevent updates or deletion.
- PRs opened with `GITHUB_TOKEN` do not trigger CI workflows on themselves.
  Arrange validation before merging the port PR, for example by running the
  documented local checks on its branch or pushing a reviewed follow-up commit
  using maintainer credentials.

## Reporting Issues

When opening an issue, include:

- What you expected to happen
- What happened instead
- Steps to reproduce
- Compiler and platform details
- Build/test command output when relevant

## Code of Conduct

This project follows the Contributor Covenant.

See `CODE_OF_CONDUCT.md` for expected behavior.
