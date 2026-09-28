# Repository upload guide

## Recommended route

Create a public GitHub repository named `CBAF-Net` or `CBAF-Net-reproducibility`. A Gitee repository is also possible if it is the stable public URL that the authors will maintain. For long-term preservation, create a tagged release and optionally archive that release with Zenodo to obtain a DOI.

## GitHub web route

1. Sign in to the author's GitHub account and create an empty repository.
2. Do not initialize it with a second README or license if those files are already present in this staging directory.
3. Copy the contents of this directory into the new repository after the final privacy, license, and source-access review.
4. Add the verified code license and environment lock file.
5. Commit the files and create an immutable tag such as `v1.0.0`.
6. Put the repository URL and tag in the manuscript's data/code availability statement and in the FCS submission form.

## Command-line route

Run these commands from the cleaned repository directory after creating the empty remote repository:

```powershell
git init
git add .
git commit -m "Initial CBAF-Net reproducibility release"
git branch -M main
git remote add origin https://github.com/<account>/CBAF-Net.git
git push -u origin main
git tag -a v1.0.0 -m "CBAF-Net reproducibility release"
git push origin v1.0.0
```

Do not put a GitHub password, personal access token, SSH private key, or container password into the manuscript, repository, or chat. Use the browser login, GitHub Desktop, or `gh auth login` locally.

## What to enter in FCS

Use the immutable release URL, for example:

`https://github.com/<account>/CBAF-Net/releases/tag/v1.0.0`

Replace the placeholder only after the authors verify the real account and tag. If the repository is not public at submission time, use the venue's allowed private/anonymous review mechanism and update the public URL before publication.
