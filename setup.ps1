# WSL2 wrapper around setup.sh.
$distro = $env:LINGBOT_WSL_DISTRO; if (-not $distro) { $distro = "Ubuntu" }
$repo   = $env:LINGBOT_REPO;       if (-not $repo)   { $repo = "~/ai/lingbot-world-v2" }
wsl -d $distro -e bash -lc "cd $repo && ./setup.sh"
exit $LASTEXITCODE
