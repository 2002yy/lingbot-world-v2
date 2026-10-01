# LingBot-World 2.0 / 1.3B causal-fast on an RTX 5060 Laptop 8GB.
#
#   .\run.ps1 play --preset performance
#   .\run.ps1 check
#
# WSL2 is assumed; this wrapper forwards into the Linux side.
param(
  [Parameter(Position=0)][string]$Command = "play",
  [string]$Preset = "performance",
  [Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest
)
$distro = $env:LINGBOT_WSL_DISTRO; if (-not $distro) { $distro = "Ubuntu" }
$repo   = $env:LINGBOT_REPO;       if (-not $repo)   { $repo = "~/ai/lingbot-world-v2" }
$inner  = "cd $repo && ./run.sh $Command --preset $Preset $($Rest -join ' ')"
wsl -d $distro -e bash -lc $inner
exit $LASTEXITCODE
