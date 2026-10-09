# Security

## Reporting

Please report a vulnerability privately through GitHub's **"Report a vulnerability"** button
(Security tab) rather than in a public issue.

## What this code trusts, and what it does not

The pipeline reads remote data from public services. Everything those services return is
treated as untrusted input:

- **Allow-listed hosts.** Asset hrefs must be `https` on Planetary Computer storage; the
  Sen1Floods11 downloader only talks to its Google Cloud Storage bucket; the CEMS client only to
  `rapidmapping.emergency.copernicus.eu` and, through at most one redirect, its S3 bucket.
- **No GDAL virtual paths or inline XML** in hrefs (`/vsi*`, `<VRTDataset>`): a catalogue item
  cannot make GDAL open arbitrary files or URLs.
- **Bounded transfers.** Every download is size-capped and time-limited; redirects are not
  followed except the one CEMS hop above.
- **Integrity.** Sen1Floods11 files are verified against the bucket's MD5 before they are written.
- **Archive extraction.** CEMS packages are extracted member by member against an allow-list of
  names; no member can be written outside its target directory.
- **XML.** Product annotations with a DTD or entity declarations are refused, and their size is capped.
- **Checkpoints** are loaded with `torch.load(weights_only=True)`.
- **Secrets.** None are needed; signed Planetary Computer URLs are redacted from logs and error
  messages.
