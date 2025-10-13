@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ---------- CONFIG ----------
set "PY=py"
set "DATA_DIR=data"
set "LOGDIR=logs"

:: Datasets (format: file|type|target)
set "OZONE=%DATA_DIR%\ozone_complet.txt|r|maxO3"
set "CARSEATS=%DATA_DIR%\Carseats.csv|c|High"

:: Algorithmes à tuner (Ridge/Lasso seront auto-sautés sur 'c')
set "ALGOS=DecisionTree RandomForest Ridge Lasso SVM"

:: n_trials par algo
set "TRIAL.DecisionTree=100"
set "TRIAL.RandomForest=20"
set "TRIAL.Ridge=1000"
set "TRIAL.Lasso=500"
set "TRIAL.SVM=250"

:: Réduire le bruit d'Optuna
set "OPTUNA_LOG_LEVEL=ERROR"

:: Horodatage + log
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "STAMP=%date:~6,4%-%date:~3,2%-%date:~0,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%"
set "STAMP=%STAMP: =0%"
set "LOG=%LOGDIR%\optuna_%STAMP%.txt"
echo Logging to "%LOG%"

(
  echo ==== START %DATE% %TIME% ====

  for %%D in ("%OZONE%" "%CARSEATS%") do (
    for /f "tokens=1-3 delims=|" %%a in ("%%~D") do (
      set "FILE=%%~a"
      set "TYP=%%~b"
      set "TARGET=%%~c"

      for %%A in (%ALGOS%) do (
        :: sauter Ridge/Lasso en classification
        set "RUN=1"
        if /I "%%A"=="Ridge" if /I "!TYP!"=="c" set "RUN=0"
        if /I "%%A"=="Lasso" if /I "!TYP!"=="c" set "RUN=0"

        :: choisir n_trials
        if /I "%%A"=="DecisionTree" (set "TRIALS=!TRIAL.DecisionTree!") ^
        else if /I "%%A"=="RandomForest" (set "TRIALS=!TRIAL.RandomForest!") ^
        else if /I "%%A"=="Ridge" (set "TRIALS=!TRIAL.Ridge!") ^
        else if /I "%%A"=="Lasso" (set "TRIALS=!TRIAL.Lasso!") ^
        else if /I "%%A"=="SVM" (set "TRIALS=!TRIAL.SVM!")

        if "!RUN!"=="1" (
          for %%L in (scratch scikit) do (
            call :run "!FILE!" "!TYP!" "!TARGET!" "%%A" "%%L" "!TRIALS!"
          )
        )
      )
    )
  )

  echo ==== END %DATE% %TIME% ====
) >> "%LOG%" 2>&1

echo Done. Log: "%LOG%"
pause
exit /b


:: ---------- SUBROUTINE ----------
:run
:: %1=file, %2=type (r/c), %3=target, %4=algo, %5=lib, %6=n_trials
echo --- %~4 (%~5) on %~nx1 [%~2 -> %~3], trials=%6 ---
"%PY%" tune_optuna.py -f "%~1" -t "%~2" -F "%~3" -a "%~4" --lib "%~5" --n-trials %6
exit /b
