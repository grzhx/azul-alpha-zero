param([ValidateSet('Release','Debug','Asan')][string]$Configuration = 'Release', [string]$OutputDirectory = '')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (!(Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio Installer/vswhere.exe not found' }
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw 'Install Visual Studio C++ desktop development tools first' }
$dev = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$build = Join-Path $root ('build-' + $Configuration.ToLowerInvariant())
if ($OutputDirectory) { $build = Join-Path $root $OutputDirectory }
New-Item -ItemType Directory -Force -Path $build | Out-Null
$flags = '/nologo /std:c++20 /EHsc /W4 /permissive- /utf-8 /MD'
if ($Configuration -eq 'Release') { $flags += ' /O2 /DNDEBUG /GL' }
elseif ($Configuration -eq 'Asan') { $flags += ' /O1 /Zi /fsanitize=address' }
else { $flags += ' /Od /Zi /RTC1' }
$include = Join-Path $root 'include'
$api = Join-Path $root 'src/c_api.cpp'
$bench = Join-Path $root 'src/benchmark.cpp'
$batchBench = Join-Path $root 'src/batch_benchmark.cpp'
$tests = Join-Path $root 'tests/test_engine.cpp'
$searchTests = Join-Path $root 'tests/test_search.cpp'
$tests3 = Join-Path $root 'tests/test_engine3.cpp'
# The generated batch file uses cmd only for compiler setup/build, never file deletion/moving.
$commands = @(
    '@echo off',
    ('call "{0}" -arch=x64 -host_arch=x64 >nul' -f $dev),
    'if errorlevel 1 exit /b 1',
    ('cl {0} /I"{1}" /DAZUL_BUILD_DLL /LD "{2}" /Fe:azul.dll /link /IMPLIB:azul.lib' -f $flags,$include,$api),
    'if errorlevel 1 exit /b 1',
    ('cl {0} /I"{1}" "{2}" /Fe:azul_bench.exe' -f $flags,$include,$bench),
    'if errorlevel 1 exit /b 1',
    ('cl {0} /I"{1}" "{2}" azul.lib /Fe:azul_batch_bench.exe' -f $flags,$include,$batchBench),
    'if errorlevel 1 exit /b 1',
    ('cl {0} /I"{1}" "{2}" azul.lib /Fe:azul_tests.exe' -f $flags,$include,$tests),
    'if errorlevel 1 exit /b 1',
    'azul_tests.exe',
    'if errorlevel 1 exit /b 1',
    ('cl {0} /I"{1}" "{2}" azul.lib /Fe:azul_search_tests.exe' -f $flags,$include,$searchTests),
    'if errorlevel 1 exit /b 1',
    'azul_search_tests.exe',
    'if errorlevel 1 exit /b 1',
    ('cl {0} /UNDEBUG /I"{1}" "{2}" azul.lib /Fe:azul_tests3.exe' -f $flags,$include,$tests3),
    'if errorlevel 1 exit /b 1',
    'azul_tests3.exe',
    'exit /b %errorlevel%'
)
$batchFile = Join-Path $build 'compile.cmd'
[IO.File]::WriteAllLines($batchFile, $commands, [Text.Encoding]::Default)
Push-Location $build
try { & cmd.exe /d /c $batchFile; if ($LASTEXITCODE -ne 0) { throw "Build/tests failed: $LASTEXITCODE" } }
finally { Pop-Location }
