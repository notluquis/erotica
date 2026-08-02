# Security policy

EROTICA is research software for astronomical catalogue analysis. It does not handle credentials,
process untrusted input from a network, or run as a service, so the realistic security surface is
small — but two classes of report are welcome and taken seriously.

## What to report

**Supply-chain and execution issues.** Anything that could cause `pip install erotica` or a normal
analysis run to execute code the user did not intend: unsafe deserialisation, `eval` on data,
path traversal when reading catalogues, or a compromised dependency pin.

**Scientific-integrity defects.** Not conventional security, but treated with the same urgency: a
bug that produces a *plausible but wrong number*. This package exists to be used in published work,
and a silently wrong result is the worst failure mode it has. Several have been found and are
documented in `docs/design-notes/decisions.md`; the class is taken seriously enough to have its own
methodology (see `methodology.md` PART K in the companion research hub).

## How

Open a [security advisory](https://github.com/notluquis/erotica/security/advisories/new) for the
first class, or a normal issue for the second — scientific defects benefit from being public, since
anyone who has used the affected version needs to know.

## Supported versions

Pre-1.0. Only the latest release is supported; there are no backports.
