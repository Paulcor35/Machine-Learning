reg='data/ozone_complet.txt'
cls='data/Carseats.csv'

./main.py -f "$reg" -s $1 -t r -F maxO3 -a DecisionTree
./main.py -f "$cls" -s $1 -t c -F High -a DecisionTree
./main.py -f "$reg" -s $1 -t r -F maxO3 -a RandomForest
./main.py -f "$cls" -s $1 -t c -F High -a RandomForest
./main.py -f "$reg" -s $1 -t r -F maxO3 -a Ridge
./main.py -f "$reg" -s $1 -t r -F maxO3 -a Lasso
./main.py -f "$reg" -s $1 -t r -F maxO3 -a SVM
./main.py -f "$cls" -s $1 -t c -F High -a SVM
