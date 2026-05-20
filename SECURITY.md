# Security Policy

We take the security of this repository and the AI/ML artifacts it produces
seriously. This policy follows the AMD-AGI Repository Standards § 2.2.3 and
the OpenSSF Best Practices Badge (Passing level).

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

You have two private reporting channels — please use whichever you prefer.

### Preferred — GitHub Private Vulnerability Reporting

This repository has GitHub
[Private Vulnerability Reporting (PVR)](https://docs.github.com/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
enabled. To report:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Fill in the advisory form with as much detail as possible.

Reports are visible only to repository administrators and the AMD-AGI security
contacts. You will receive an acknowledgement within 5 business days.

### Fallback — AMD PSIRT email

If GitHub PVR is unavailable to you, email the AMD Product Security Incident
Response Team:

- **psirt@amd.com**

Please include:

- Repository name and affected versions / commit SHA
- A description of the issue and its potential impact
- Steps to reproduce, including any required configuration or inputs
- Suggested mitigation, if any
- Whether you wish to be credited in the advisory

## What to Report

Examples of issues to report privately:

- Remote code execution, sandbox escapes, privilege escalation
- Authentication or authorization bypass
- Leakage of credentials, tokens, or proprietary model weights
- Malicious model artifacts (data poisoning, backdoored checkpoints,
  adversarial-input vulnerabilities specific to AI/ML pipelines)
- Dependency vulnerabilities with no upstream advisory yet

## Coordinated Disclosure

We follow coordinated disclosure. Once a fix is available we will:

1. Publish a GitHub Security Advisory describing the issue, affected versions,
   and mitigations.
2. Credit the reporter (with consent).
3. Tag a release containing the fix.

We aim to resolve confirmed vulnerabilities within 90 days of triage; critical
issues are prioritized for faster resolution.

## Out of Scope

- Bugs without a security impact — please open a normal issue.
- Findings from automated scanners without a working proof of concept.
- Issues that require physical access to a developer machine.

## Supported Versions

Unless explicitly stated in the README, only the default branch (`main`)
receives security updates.

Thank you for helping keep AMD-AGI projects and their users safe.
