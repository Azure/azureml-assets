FROM mcr.microsoft.com/dotnet/framework/runtime:3.5-windowsservercore-ltsc2019

# # Install VC 10, VC 12 and VC14
# RUN powershell.exe -Command \
#     wget https://download.visualstudio.microsoft.com/download/pr/10912041/cee5d6bca2ddbcd039da727bf4acb48a/vcredist_x64.exe -OutFile $env:temp\\vc_redist.x64.exe; \
#     Start-Process $env:temp\\vc_redist.x64.exe -ArgumentList "/quiet", "/install" -Wait; \
#     Remove-Item -Force $env:temp\\vc_redist.x64.exe; \
#     echo "VC12-INSTALLED"; \
#     wget https://aka.ms/vs/16/release/VC_redist.x64.exe -OutFile:$env:temp\\VC_64.exe;\
#     Start-Process $env:temp\\vc_64.exe -ArgumentList "/quiet", "/install" -Wait; \
#     Remove-Item -Force $env:temp\\vc_64.exe; \
#     echo "VC14-INSTALLED"

RUN powershell.exe -Command \
    echo "INSTALLING-CONDA"; \    
    echo "START-DOWNLOAD"; \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Windows-x86_64.exe -OutFile $env:temp\\miniconda.exe; \
    echo "DONE-DOWNLOAD"; \
    echo "START-INSTALL"; \
    Start-Process $env:temp\\miniconda.exe -ArgumentList "/S", "/InstallationType=AllUsers", "/AddToPath=1", "/D=$env:LOCALAPPDATA\\Miniconda3" -Wait; \
    echo "DONE-INSTALL"; \
    echo "REMOVE-TEMP"; \
    Remove-Item $env:temp\\miniconda.exe -Force; \
    echo "CONDA-INSTALLED";