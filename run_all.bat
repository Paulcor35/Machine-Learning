@echo off
REM Usage : run_all.bat all

set ARG=%1

py main.py -f data/ozone_complet.txt -s %ARG% -t r -F maxO3 -a DecisionTree
py main.py -f data/Carseats.csv -s %ARG% -t c -F High -a DecisionTree
py main.py -f data/ozone_complet.txt -s %ARG% -t r -F maxO3 -a RandomForest
py main.py -f data/Carseats.csv -s %ARG% -t c -F High -a RandomForest
py main.py -f data/ozone_complet.txt -s %ARG% -t r -F maxO3 -a Ridge
py main.py -f data/ozone_complet.txt -s %ARG% -t r -F maxO3 -a Lasso
py main.py -f data/ozone_complet.txt -s %ARG% -t r -F maxO3 -a SVM
py main.py -f data/Carseats.csv -s %ARG% -t c -F High -a SVM

pause
