# Security Policy

## Supported Versions

<!-- TODO(gate-4): confirm this table at the v0.1.0 tag. PRISM has no tagged
     release yet; the row below states the intended post-release policy. -->

| Version | Supported |
|---|---|
| `main` (unreleased) | ✅ Current development branch |
| 0.1.x | ⏳ Supported once tagged |
| < 0.1.0 | ❌ No releases exist prior to 0.1.0 |

Until v0.1.0 is tagged, only the current `main` branch receives fixes.

## Reporting a Vulnerability

**Do not open a public issue for a suspected vulnerability or an exposed
secret.**

Use GitHub's [private vulnerability reporting][pvr] for this repository
(Security → Report a vulnerability). If that is unavailable to you, email
<smadireddy@anl.gov> directly.

[pvr]: https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability

Please include:

- the affected revision and component;
- reproduction steps or a proof of concept;
- the expected impact; and
- any known mitigation.

### What to Expect

- **Acknowledgment** within five business days.
- Validation and remediation coordinated privately with you.
- Credit in the advisory if you would like it and it is legally possible.

Please give us a reasonable opportunity to address the issue before any public
disclosure.

## Scope

PRISM is a research training framework for scientific multimodal models. The
following are in scope:

- Credential, token, or key exposure in the repository, its history, or in
  generated launcher and job scripts
- Code execution paths reachable from untrusted model configs, checkpoints, or
  dataset manifests
- Deserialization of untrusted checkpoints
- CI workflow vulnerabilities, including those reachable from fork pull
  requests

The following are generally **out of scope**:

- Vulnerabilities in third-party dependencies — report those upstream, though
  we welcome a heads-up
- Attacks requiring privileged access to an HPC allocation you are already
  authorized to use
- Model behavior concerns such as hallucination, bias, or output quality, which
  are research issues rather than security vulnerabilities — please open a
  normal issue for those

## Sensitive Data

Never commit access tokens, private keys, scheduler credentials, personal data,
unpublished datasets, private filesystem paths, or credentials embedded in
generated launch scripts.

**If a secret is committed, rotate it immediately.** Removing it from the
latest revision is not sufficient — the value remains recoverable from Git
history and must be treated as compromised from the moment it was pushed.

Absolute paths under `/home/<user>`, `/flare`, `/lus`, `/eagle`, or
`/global/homes` should not appear in tracked files. Pre-existing occurrences
are being removed as part of release readiness; new ones should not be added.

## Model and Data Provenance

PRISM redistributes third-party tokenizer assets and can load third-party
pretrained weights. Attributions and their licenses are recorded in
[`NOTICE`](./NOTICE). If you believe an asset is misattributed or
redistributed without the right license, report it through the same private
channel above.

Treat checkpoints from untrusted sources as untrusted input: loading a
pickle-backed checkpoint can execute arbitrary code.
