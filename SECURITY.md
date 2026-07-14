# Security Policy

## Supported versions

autotrainer is pre-1.0 (SemVer 0.x). Security fixes are applied only to the
latest released version on PyPI; there is no backport policy for older lines
yet.

| Version | Supported          |
|---------|--------------------|
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Instead, report vulnerabilities privately so a fix can be prepared before
details are public:

1. Go to **https://github.com/OriAlpha/autotrainer/security/advisories/new**
   and create a private security advisory, **or**
2. Email **suhasgoravale@gmail.com** with the details.

Please include:

- A description of the issue and its potential impact.
- Steps to reproduce, or a proof-of-concept.
- The version(s) of autotrainer affected.
- The output of `autotrainer doctor` and `autotrainer info` if relevant.

## Response timeline

- **Acknowledgement**: within 72 hours.
- **Initial assessment**: within 7 days.
- **Fix or mitigation**: target 30 days, depending on severity and complexity.
- A CVE will be requested where appropriate once a fix is available.

We ask that you give us reasonable time to respond before any public
disclosure. We will credit reporters in the release notes unless you prefer
to remain anonymous.
