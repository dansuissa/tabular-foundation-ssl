# First private GitHub publication

The repository is staged without credentials or licensed weights. The included
PowerShell script creates the private GitHub repository under the authenticated
account, pushes the reviewed research record, adds repository topics, and
uploads the checksummed raw cluster archive as a private release asset.

From the extracted bundle on the authenticated Windows laptop:

```powershell
cd .\tabular-foundation-ssl
.\publish_private_repo.ps1
```

The script defaults to `dansuissa/tabular-foundation-ssl` and refuses to proceed
if that repository already exists. It refuses to publish if the dependency-free
repository verification fails; when a compatible local test environment is
already installed, it also reruns the test suite. No token is read from a file
or placed in a command; authentication is delegated to the existing GitHub CLI
keyring session.

After publication:

1. Open the repository page and confirm it displays **Private**.
2. Confirm the README image, complete ranking, and PDF links render.
3. Confirm the `repository validation` workflow is green.
4. Open release `research-record-v1` and verify the artifact checksum shown in
   `docs/ARTIFACTS.md`.
5. Invite lab collaborators from **Settings → Collaborators and teams** using
   their exact GitHub accounts. Do not make the repository public to simplify
   access.
