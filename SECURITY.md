# Security

## Reporting

Please report a vulnerability privately through GitHub's **"Report a vulnerability"** button
(Security tab) rather than in a public issue.

## What this code trusts, and what it does not

The pipeline reads remote data from public services. Everything those services return is
treated as untrusted input (`src/floodfusion/data/http.py` holds the guards; `tests/test_http.py`
pins each bypass a security review demonstrated):

- **Allow-listed hosts, checked by the parser that makes the request.** A URL passes only if
  `urllib.parse` and urllib3 (what `requests` uses) agree on its host, it carries no user-info,
  backslash, whitespace or control character, and its port is the default. Asset hrefs must be
  `https` on Planetary Computer storage; the Sen1Floods11 downloader only talks to its Google
  Cloud Storage bucket path; the CEMS client only to `rapidmapping.emergency.copernicus.eu` and,
  through at most one validated redirect, its S3 bucket.
- **Local paths given to GDAL are plain paths.** `/vsi*` in any case, driver or scheme prefixes
  (`GTIFF_DIR:`, `file:` other than `file:///abs`) and inline XML are refused, and GeoTIFFs are
  opened with the `GTiff` driver only — a downloaded `.tif` holding VRT XML cannot reference
  other files or URLs.
- **Bounded transfers.** Every download is size-capped and time-limited (GDAL's own HTTP reads
  too, in every worker thread); redirects are not followed except the one CEMS hop; transient
  failures are retried with backoff, permanent ones are not.
- **Integrity.** Sen1Floods11 files are verified against the bucket's MD5 before they are
  written (this protects against corruption, not against a compromised bucket).
- **Archive extraction.** CEMS packages are extracted member by member against a full-match,
  ASCII-only allow-list of names, at most one package-named folder deep, with per-member,
  member-count and total-size caps, streamed to disk; no member can be written outside its
  target directory. API strings that become directory names are validated first.
- **XML.** Product annotations with a DTD or entity declarations are refused, and their size is capped.
- **Checkpoints** are loaded with `torch.load(weights_only=True)`.
- **Secrets.** None are needed. Signed Planetary Computer URLs are redacted from logs, error
  messages and tracebacks, including urllib3's `url: /path?…` form.
- **Supply chain.** CI actions and Docker base images are pinned by digest; Dependabot proposes
  updates for actions, the uv lock and the images.
