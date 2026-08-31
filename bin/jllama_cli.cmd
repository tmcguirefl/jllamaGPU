@echo off
setlocal EnableExtensions
rem Convenience wrapper: run repo-root jllama_cli.ijs via jconsole.
rem Honors JCONSOLE if set. Does not use the #! shebang (Windows cannot).

set "ROOT=%~dp0.."
set "CLI=%ROOT%\jllama_cli.ijs"

if not exist "%CLI%" (
  echo jllama_cli: missing %CLI% 1>&2
  exit /b 127
)

if not defined JCONSOLE (
  if exist "C:\Users\tmcguire\j9.8\bin\jconsole.exe" set "JCONSOLE=C:\Users\tmcguire\j9.8\bin\jconsole.exe"
)
if not defined JCONSOLE set "JCONSOLE=jconsole.exe"

rem Blank line so jconsole does not wait on "Press ENTER to inspect" after errors.
echo.| "%JCONSOLE%" "%CLI%" %*
exit /b %ERRORLEVEL%
