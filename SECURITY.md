# Security policy

## Reporting a problem

Please report security issues privately through GitHub's
[security advisories](https://github.com/mohammadazam7/ml-research-agent/security/advisories/new)
rather than in a public issue. Expect an acknowledgement within a few days.

## How the agent is constrained

The agent writes and runs training code, so the design assumes that code can be wrong or
manipulated:

- Generated code runs in a separate process with no network access, a time limit, a
  memory limit and a working directory it cannot leave.
- It can read only the dataset it was given and write only to its run directory.
- Column names and values in the data are treated as data, never as instructions.
- API keys for model providers are read from the environment and are never committed.

Reports of a way to escape the sandbox, reach the network or read outside the dataset
are especially welcome.
