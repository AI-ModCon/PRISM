## Description

Summarize the change and the motivation behind it. Describe the user-visible
behavior, not just the implementation.

Fixes # (issue number)

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that changes existing behavior)
- [ ] Documentation update
- [ ] Infrastructure, CI, or tooling

## Compatibility

PRISM has downstream users running long jobs on HPC systems. Does this PR
change any of the following?

- [ ] `src.*` import paths
- [ ] the `prism` CLI surface
- [ ] Hydra config keys
- [ ] checkpoint keys
- [ ] launcher flags
- [ ] None of the above

If any box above is checked, describe the migration path here.

## How Has This Been Tested?

List what you ran, and what you intentionally skipped and why. Hardware-specific
results may be reported here when the required platform is unavailable to you.

```
# e.g.
bash tools/ci/local_quality.sh full
pytest -q -m "not multimodal and not launcher and not integration and not network and not slow and not gpu and not aurora and not perlmutter" tests
```

- Platform tested on:
- Tests skipped, and why:

## New Dependencies, Data, or Assets

Any new dependency, dataset, pretrained model, tokenizer, or other asset — with
its license and provenance. Write "None" if there are none.

## Checklist

- [ ] I have performed a self-review of my own changes
- [ ] The code follows the project's style guidelines (`bash tools/ci/local_quality.sh`)
- [ ] I have commented my code where the intent is not obvious
- [ ] I have made corresponding changes to the documentation
- [ ] I have added tests that cover this change, or explained why none are needed
- [ ] New and existing tests pass locally
- [ ] No secrets, credentials, or absolute institutional paths are included
      (see [SECURITY.md](https://github.com/AI-ModCon/BaseMM_PRISM/blob/main/SECURITY.md))

## AI/LLM Assistance

See *Guidelines for AI/LLM-Assisted Contributions* in
[CONTRIBUTING.md](https://github.com/AI-ModCon/BaseMM_PRISM/blob/main/CONTRIBUTING.md). All PRs require a human reviewer who is
accountable for the review — a project norm, not yet enforced by branch
protection.

- [ ] AI/LLM tools were primarily used to generate code or artifacts in this PR
- [ ] I have reviewed all such output line-by-line and understand it

## Additional Context

Anything else a reviewer should know.
