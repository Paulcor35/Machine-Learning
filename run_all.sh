./main.py -f data/ozone_complet.txt -s $1 -t r -F maxO3 -a DecisionTree
./main.py -f data/Carseats_prepared.csv -s $1 -t c -F High -a DecisionTree
./main.py -f data/ozone_complet.txt -s $1 -t r -F maxO3 -a RandomForest
./main.py -f data/Carseats_prepared.csv -s $1 -t c -F High -a RandomForest
./main.py -f data/ozone_complet.txt -s $1 -t r -F maxO3 -a Ridge
./main.py -f data/ozone_complet.txt -s $1 -t r -F maxO3 -a Lasso
./main.py -f data/ozone_complet.txt -s $1 -t r -F maxO3 -a SVM
./main.py -f data/Carseats_prepared.csv -s $1 -t c -F High -a SVM
