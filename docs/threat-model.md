# Threat model (STRIDE, lightweight)

Scope: a local, offline test library and its CI. No services, no user data, no model weights downloaded, no credentials.

| STRIDE | Threat | Control |
|---|---|---|
| Spoofing | Malicious GitHub Action impersonating a trusted one | Actions pinned to full commit SHAs |
| Tampering | Dependency compromise (torch, pytest, ruff) | Pinned versions in `requirements-ci.txt`; Dependabot alerts; majors reviewed by hand |
| Repudiation | Unverifiable README claims | Docs check regenerates the test count and every captured output in CI |
| Information disclosure | Secrets or real organisation names committed | gitleaks on full history; hashed brand blocklist check |
| Denial of service | CI hangs | `timeout-minutes: 20`; tests run in about 2 s |
| Elevation of privilege | Workflow token misuse | `permissions: contents: read` (CodeQL adds only `security-events: write`) |

OWASP LLM Top 10: mostly not applicable (no LLM application, prompts or tools). The most relevant is LLM05/Supply chain: when adopting the guide with real open-weights models, load weights from trusted sources and prefer safetensors over pickle.
