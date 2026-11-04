# escape=`
# Windows image approximating the GitHub windows-2022 runner for scripts/docker-ci.ps1.
FROM mcr.microsoft.com/windows/servercore:ltsc2022
SHELL ["powershell", "-NoProfile", "-Command", "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue';"]

RUN Set-ExecutionPolicy Bypass -Scope Process -Force; `
    [Net.ServicePointManager]::SecurityProtocol = 3072; `
    iex ((New-Object Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
RUN choco install -y --no-progress git ninja llvm; `
    choco install -y --no-progress cmake --installargs 'ADD_CMAKE_TO_PATH=System'

RUN Invoke-WebRequest https://win.rustup.rs/x86_64 -OutFile C:\rustup-init.exe; `
    C:\rustup-init.exe -y --default-toolchain stable --profile minimal; `
    Remove-Item C:\rustup-init.exe

# MSVC with the Windows 11 SDK 22621; 26100 is removed so the SDK matches the runner.
RUN Invoke-WebRequest https://aka.ms/vs/17/release/vs_buildtools.exe -OutFile C:\vs_buildtools.exe; `
    $p = Start-Process C:\vs_buildtools.exe -Wait -PassThru -ArgumentList '--quiet','--wait','--norestart','--nocache', `
        '--installPath','C:\BuildTools','--add','Microsoft.VisualStudio.Workload.VCTools','--includeRecommended', `
        '--add','Microsoft.VisualStudio.Component.Windows11SDK.22621'; `
    if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) { throw ('vs_buildtools exit ' + $p.ExitCode) }; `
    Remove-Item C:\vs_buildtools.exe; `
    $k = 'C:\Program Files (x86)\Windows Kits\10'; `
    foreach ($d in 'bin','Include','Lib','Redist','UnionMetadata','References') { `
        $p = Join-Path (Join-Path $k $d) '10.0.26100.0'; if (Test-Path $p) { Remove-Item -Recurse -Force $p } }

# The workflow steps run under `shell: pwsh`. 7.4 LTS matches the runner; choco's 7.6.9
# crashes on module autoload (e.g. Expand-Archive) inside Windows containers.
RUN Invoke-WebRequest https://github.com/PowerShell/PowerShell/releases/download/v7.4.6/PowerShell-7.4.6-win-x64.zip -OutFile C:\pwsh.zip; `
    Expand-Archive C:\pwsh.zip C:\pwsh; Remove-Item C:\pwsh.zip; `
    [Environment]::SetEnvironmentVariable('PATH', 'C:\pwsh;' + [Environment]::GetEnvironmentVariable('PATH', 'Machine') + `
        ';C:\Users\ContainerAdministrator\.cargo\bin;C:\Program Files\LLVM\bin', 'Machine')
ENV LIBCLANG_PATH="C:\Program Files\LLVM\bin"

COPY run-ci.ps1 C:/ci/run-ci.ps1
WORKDIR C:/work
