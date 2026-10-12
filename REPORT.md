# SIECI NEURONOWE – ćwiczenie

Implementacja modelu regresji liniowej na zbiorze Heart Disease

Repozytorium: [https://github.com/mateusz-kow/neural-networks-list-2](https://github.com/mateusz-kow/neural-networks-list-2)

Autor: Mateusz Kowalski 280543

---

### Wprowadzenie do zbioru Heart Disease

W ćwiczeniu wykorzystałem zbiór danych Heart Disease udostępniony w repozytorium UCI Machine Learning Repository. Zbiór zawiera informacje o pacjentach, obejmujące cechy demograficzne oraz wyniki wybranych badań medycznych.

Zbiór zawiera 303 obserwacje oraz 13 cech wejściowych. Pierwotna zmienna docelowa num przyjmuje wartości od 0 do 4, gdzie wartość 0 oznacza brak rozpoznanej choroby serca, a wartości większe od 0 wskazują na jej obecność.

| Nazwa cechy | Opis | Typ w analizie |
| ----- | ----- | ----- |
| `age` | Wiek pacjenta w latach | Numeryczna |
| `sex` | Płeć pacjenta (0 lub 1\) | Kategoryczna |
| `cp` | Rodzaj bólu w klatce piersiowej | Kategoryczna |
| `trestbps` | Ciśnienie krwi w spoczynku (mm Hg) | Numeryczna |
| `chol` | Poziom cholesterolu w surowicy (mg/dl) | Numeryczna |
| `fbs` | Informacja, czy glikemia na czczo przekracza 120 mg/dl | Kategoryczna |
| `restecg` | Wynik elektrokardiogramu w spoczynku | Kategoryczna |
| `thalach` | Maksymalna osiągnięta częstość akcji serca | Numeryczna |
| `exang` | Występowanie dławicy piersiowej wywołanej wysiłkiem | Kategoryczna |
| `oldpeak` | Obniżenie odcinka ST wywołane wysiłkiem względem spoczynku | Numeryczna |
| `slope` | Nachylenie szczytowego odcinka ST podczas wysiłku | Kategoryczna |
| `ca` | Liczba głównych naczyń uwidocznionych w badaniu fluoroskopowym | Numeryczna |
| `thal` | Wynik badania dotyczącego talasemii, zakodowany kategoriami diagnostycznymi | Kategoryczna |
| `num` | Pierwotna zmienna docelowa określająca wynik diagnostyczny | Docelowa |

### Przygotowanie danych do uczenia

Następnym krokiem było przygotowanie danych do klasyfikacji. W tym celu użyłem pakietu scikit-learn. Cechy numeryczne standaryzowałem do porównywalnej skali przy użyciu StandardScalera, natomiast dla cech kategorycznych użyłem OneHotEncodera.

Zbiór podzieliłem na 80% danych treningowych i 20% danych testowych. Na tym etapie wykorzystałem stratyfikację, aby osiągnąć porównywalne proporcje klas w obu podzbiorach.

### Baseline klasyfikatora

Jako klasyfikator bazowy zastosowałem regresję logistyczną z biblioteki scikit-learn. Model wytrenowałem na przygotowanym zbiorze treningowym, a jego skuteczność oceniłem na zbiorze testowym.

| Metryka | Wynik |
| ----- | ----- |
| Accuracy | 86,9% |
| F1-score dla klasy 1 | 0,867 |

| Klasa rzeczywista / przewidywana | 0 | 1 |
| ----- | ----- | ----- |
| 0 | 27 | 6 |
| 1 | 2 | 26 |

Model poprawnie sklasyfikował 53 z 61 obserwacji testowych.

### Implementacja modelu regresji liniowej

### Metryki jakości modelu

### Wnioski
