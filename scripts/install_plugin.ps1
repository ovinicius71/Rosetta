# Instala/atualiza o plugin Rosetta no Xournal++ (copia para a pasta de plugins do usuário).
# Rodar de qualquer lugar; reexecutar a cada mudança no plugin (Xournal++ carrega no boot).

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$src = Join-Path $repo "xournalpp-plugin\rosetta"
$dst = Join-Path $env:LOCALAPPDATA "xournalpp\plugins\rosetta"

if (-not (Test-Path $src)) { throw "plugin não encontrado em $src" }

New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$src\*" $dst -Recurse -Force

Write-Host "Plugin instalado em $dst"
Write-Host "Reinicie o Xournal++ e use Plugin > 'Rosetta: reconhecer contas' (Ctrl+M)."
