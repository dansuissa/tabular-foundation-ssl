param(
    [string]$Owner = "dansuissa",
    [string]$Repository = "tabular-foundation-ssl"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$FullName = "$Owner/$Repository"
$Description = "Reproducible benchmark of tabular foundation models and semi-supervised learning under label scarcity"
$Artifact = Join-Path (Split-Path $PSScriptRoot -Parent) "release_assets\tabular-foundation-ssl-full-cluster-artifacts.tar.gz"
$Checksum = "${Artifact}.sha256"

Write-Host "Checking GitHub authentication..."
gh auth status
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI authentication is not active. Run 'gh auth login' first."
}

gh repo view $FullName *> $null
if ($LASTEXITCODE -eq 0) {
    throw "Repository $FullName already exists. Refusing to overwrite or push into an unknown remote."
}

Write-Host "Validating the complete research record..."
python scripts/verify_repository.py
if ($LASTEXITCODE -ne 0) { throw "Repository verification failed." }

python -c "import pytest" 2>$null
if ($LASTEXITCODE -eq 0) {
    python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Test suite failed." }
} else {
    Write-Warning "pytest is not installed in this shell; using the committed validated test record."
}

if (-not (Test-Path ".git")) {
    git init -b main
}

if (-not (git config --get user.name)) {
    $GitHubName = gh api user --jq '.name // .login'
    git config user.name $GitHubName
}
if (-not (git config --get user.email)) {
    $GitHubLogin = gh api user --jq '.login'
    $GitHubId = gh api user --jq '.id'
    git config user.email "$GitHubId+$GitHubLogin@users.noreply.github.com"
}

git add --all
git status --short
git commit -m "Initial private research release"
if ($LASTEXITCODE -ne 0) { throw "Initial Git commit failed." }

Write-Host "Creating the private repository and pushing main..."
gh repo create $FullName --private --description $Description --source . --remote origin --push
if ($LASTEXITCODE -ne 0) { throw "GitHub repository creation or push failed." }

gh repo edit $FullName `
    --add-topic tabular-data `
    --add-topic semi-supervised-learning `
    --add-topic foundation-models `
    --add-topic openml `
    --add-topic reproducible-research

if (Test-Path $Artifact) {
    Write-Host "Uploading the checksummed cluster artifact archive..."
    gh release create research-record-v1 $Artifact `
        $Checksum `
        --repo $FullName `
        --title "Complete research record" `
        --notes "Canonical code, results, reports, and the checksummed raw Slurm shard/log archive for the completed tabular foundation-model and SSL benchmark."
    if ($LASTEXITCODE -ne 0) { throw "Release asset upload failed." }
} else {
    Write-Warning "Cluster artifact archive not found at $Artifact; the repository was pushed without the release asset."
}

Write-Host "Publication complete: https://github.com/$FullName"
