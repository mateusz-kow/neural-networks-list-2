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
| Accuracy | 0.869 |
| F1-score dla klasy 1 | 0,867 |

| Klasa rzeczywista / przewidywana | 0 | 1 |
| ----- | ----- | ----- |
| 0 | 27 | 6 |
| 1 | 2 | 26 |

Model poprawnie sklasyfikował 53 z 61 obserwacji testowych.

### Implementacja modelu regresji liniowej

Zaimplementowałem własny model regresji logistycznej od zera w bibliotece NumPy. Wykorzystałem:
* **Funkcję aktywacji:** Sigmoida sigma(z) = 1 / (1 + e^(-z)).
* **Funkcję kosztu:** Entropię krzyżową (Binary Cross-Entropy): L = -y * ln p(x) - (1-y) * ln(1-p(x)).
* **Optymalizator:** Gradient prosty (Batch Gradient Descent) z regułą aktualizacji wag opisanej w poleceniu

Podczas finalizacji rozwiązania pokazałem własną implementację LLM'owi i ten mnie uświadomił, że mozliwa jest optymalizacja, polegająca na wykorzystaniu zachowania biblioteki numpy dotyczącej mnożenia macierzy, wektorów i skalarów. Wobec tego zamiast iterować po każdym pacjencie z osobna w kodzie została utworzona macierz pacjentów, która jest mnożona przez wektor wag zwracając tym samym wektor prawdopodobieństw.

### Metryki jakości modelu

Przeprowadziłem trzy serie eksperymentów badawczych w celu zbadania wpływu hiperparametrów na zbieżność i jakość klasyfikacji.

![Learning curve alpha F1](plots/learning_curve_alpha_f1.png)
![Learning curve alpha loss](plots/learning_curve_alpha_loss.png)

#### 1. Wpływ współczynnika uczenia (alpha)
Badanie przeprowadzono dla liczby epok = 1000 przy inicjalizacji zerami:
* alpha = 0.50 | Final Loss: 0.3349 | Test F1: 0.847
* alpha = 0.10 | Final Loss: 0.3368 | Test F1: 0.867
* alpha = 0.01 | Final Loss: 0.3739 | Test F1: 0.881
* alpha = 0.001 | Final Loss: 0.5369 | Test F1: 0.857
* alpha = 0.0001 | Final Loss: 0.6688 | Test F1: 0.857

![Learning curve init loss](plots/learning_curve_init_loss.png)

#### 2. Wpływ rozkładu inicjalizacji parametrów
Dla alpha = 0.1 i 1000 epok:
* **Zeros:** Final Loss: 0.3368 | Test F1: 0.867
* **Normal (std=0.01):** Final Loss: 0.3368 | Test F1: 0.867
* **Uniform (-0.01, 0.01):** Final Loss: 0.3368 | Test F1: 0.867
* **Large Normal (std=1.0):** Final Loss: 0.3374 | Test F1: 0.862

#### 3. Ostateczne wyniki najlepszego modelu
Najlepszy model osiągnięto przy konfiguracji: alpha = 0.1, inicjalizacja zerami, liczba epok = 2000.
* **Accuracy:** 0.869
* **Precision:** 0.812
* **Recall:** 0.929
* **F1-score:** 0.867

### Wnioski

1. **Czy klasyfikator liniowy zawsze zbiega do pożądanego przez nas globalnego minimum?**

Tak, pod warunkiem, że współczynnik alpha jest większa od zera. Wynika to z matematycznych właściwości wykorzystanych funkcji.

2. **Czy mogą jednak pojawić się problemy związane ze złą parametryzacją i czy te problemy kompletnie uniemożliwiają zbieżność/dobrą jakość wyniku, czy tylko przedłużają uczenie?**

Problemy związane ze złą parametryzacją jak najbardziej mogą wystąpić i mogą mieć wpływ zarówno na jakość wyniku jak i na szybkość jego trenowania.

Dla małych wartości współczynnika alpha zaobserwowano, iż wytrenowany model charakteryzuje duża niepewność (loss), co jest skutkiem tego, że w zaprezentowanym wyżej (na wykresach) przypadku model był trenowany na 1000 epok, co nie było wystarczające. Rozwiązaniem tego problemu mogłoby być zwiększenie liczby epok, co z kolei wiązałoby się z wydłużonym czasem tenowania modelu.

Dla dużych wartości współczynnika alpha natomiast dokonano obserwacji sugerującej, że model bardzo szybko znalazł się blisko optimum, ale nie mógł nigdy do niego dotrzeć, ponieważ zawsze je 'przeskakiwał'. Przez to taki model może nigdy nie znaleźć równie odpowiednich parametrów co model o mniejszym współczynniku alpha.

Z kolei jeżeli chodzi o początkowe wartości parametrów W, to szczególną różnicę robią one w pierwszych epokach uczenia, natomiast wraz z kolejnymi epokami kolejne wartości parametrów modelu zbiegają się do siebie.

Podsumowując, trenując model, najlepiej wybrać taki współczynnik alpha, który zapewnia kompromis pomiędzy szybkością zbieżności a stabilnością optymalizacji (w naszym przypadku alpha = 0.1 lub 0.01).

---

Wykorzystanie generatywnej sztucznej inteligencji w ramach zadania ogranicało się głównie do wygenerowania kodu tworzącego wykresy oraz liczącego metryki a także optymalizacji wykorzystania biblioteki numpy.
