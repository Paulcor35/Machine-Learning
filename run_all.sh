./main.py -f Data-20251001/ozone_complet.txt -t r -F maxO3 -a DecisionTree
./main.py -f Data-20251001/Carseats_prepared.csv -t c -F High -a DecisionTree
./main.py -f Data-20251001/ozone_complet.txt -t r -F maxO3 -a RandomForest
./main.py -f Data-20251001/Carseats_prepared.csv -t c -F High -a RandomForest
./main.py -f Data-20251001/ozone_complet.txt -t r -F maxO3 -a Ridge
./main.py -f Data-20251001/ozone_complet.txt -t r -F maxO3 -a Lasso
./main.py -f Data-20251001/ozone_complet.txt -t r -F maxO3 -a SVM
./main.py -f Data-20251001/Carseats_prepared.csv -t c -F High -a SVM
