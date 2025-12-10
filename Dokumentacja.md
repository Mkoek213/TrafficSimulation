# Dokumentacja Symulacji Ruchu Drogowego (Site A)

## 1. Wstęp

Niniejszy projekt stanowi symulację systemu dyskretnego ruchu drogowego, modelującą zachowanie pojazdów na skrzyżowaniu typu "Site A" (na podstawie datasetu DRIFT). Symulacja została stworzona w celu analizy przepustowości skrzyżowania, identyfikacji punktów zatorowych oraz badania bezpieczeństwa ruchu poprzez analizę zdarzeń hamowania i przyspieszania.

### Cel i Przypadki Użycia (Use Cases)
Głównym celem symulacji jest stworzenie wirtualnego środowiska odzwierciedlającego rzeczywiste warunki drogowe, co pozwala na:
1.  **Optymalizację Infastruktury**: Testowanie wpływu zmian w organizacji ruchu (np. zmiana liczby pasów) na przepustowość.
2.  **Badanie Algorytmów Autonomicznych**: Wykorzystanie modelu jako środowiska testowego dla algorytmów sterowania pojazdami autonomicznymi, aby zapewnić bezpieczne i płynne włączanie się do ruchu.
3.  **Analizę Bezpieczeństwa**: Identyfikacja miejsc o podwyższonym ryzyku kolizji poprzez analizę gwałtownych hamowań (deceleracji).

---

## 2. Modele Matematyczne

Symulacja opiera się na dwóch kluczowych modelach opisujących dynamikę pojazdów: modelu podążania za liderem (Car-Following) oraz modelu zmiany pasa ruchu (Lane-Changing).

### 2.1. Model Kraussa (Car-Following)
Zaimplementowany w `src/utils/krauss_model.py`.
Model Kraussa jest stochastycznym modelem podążania za pojazdem poprzedzającym, który wyznacza bezpieczną prędkość, uwzględniając czas reakcji kierowcy oraz ograniczenia fizyczne pojazdu.

**Kluczowe założenia:**
*   Pojazd dąży do osiągnięcia prędkości maksymalnej ($v_{max}$), ale musi zachować bezpieczny odstęp od lidera.
*   Uwzględniana jest losowa deceleracja ($\epsilon$), symulująca niedoskonałości ludzkiego sterowania.

**Równania:**
1.  **Bezpieczna prędkość ($v_{safe}$):**
    Wyliczana tak, aby uniknąć kolizji w przypadku nagłego hamowania lidera.
    $$v_{safe} = v_l + \frac{g - g_{des}}{\tau}$$
    Gdzie:
    *   $v_l$ - prędkość lidera
    *   $g$ - aktualny odstęp (gap)
    *   $g_{des}$ - pożądany odstęp
    *   $\tau$ - czas reakcji kierowcy (reaction_time)

2.  **Pożądany odstęp ($g_{des}$):**
    $$g_{des} = g_{min} + \max\left(0, \frac{v_f (v_f/a_{max} + \tau) - v_l^2/a_{max}}{2}\right) \cdot 1.2$$
    (Wzór analityczny zapewniający margines bezpieczeństwa).

3.  **Prędkość w kolejnym kroku ($v_{next}$):**
    $$v_{des} = \min(v_{max}, v_{safe}, v + a_{max} \cdot \Delta t)$$
    $$v_{next} = \max(0, v_{des} - \epsilon)$$
    Gdzie $\epsilon$ to losowe przyhamowanie (random deceleration).

### 2.2. Model MOBIL (Lane-Changing)
Zaimplementowany w `src/utils/mobil_model.py`.
MOBIL (Minimizing Overall Braking Induced by Lane changes) to model decyzyjny dla zmiany pasa ruchu. Decyzja podejmowana jest na podstawie "zachęty" (incentive) oraz kryterium bezpieczeństwa.

**Kryteria:**
1.  **Kryterium Bezpieczeństwa:**
    Zmiana pasa jest możliwa tylko wtedy, gdy nie wymusi na pojeździe nadjeżdżającym z tyłu (na nowym pasie) hamowania przekraczającego bezpieczny próg ($b_{safe}$).

2.  **Kryterium Zachęty (Incentive):**
    Pojazd zmieni pas, jeśli zysk własny w przyspieszeniu przewyższa (zważony czynnikiem "uprzejmości" $p$) straty przyspieszenia innych pojazdów.

    $$ \tilde{a}_c - a_c + p (\tilde{a}_n - a_n) > \Delta a_{th} $$

    W implementacji:
    *   `accel_gain` = $\tilde{a}_c - a_c$ (Zysk pojazdu zmieniającego pas)
    *   `follower_impact` = $a_n - \tilde{a}_n$ (Strata pojazdu podążającego - *Note: w kodzie zdefiniowane jako różnica przed i po*)
    *   `total_incentive` = `accel_gain` - $p \cdot$ `follower_impact` $\pm$ `right_lane_bias`

---

## 3. Analiza Danych (Dataset DRIFT)

Projekt wykorzystuje dane z datasetu DRIFT (dane z dronów, Site A), przetworzone w `eda/notebooks/car_density.ipynb`. Dataset zawiera trajektorie pojazdów (współrzędne $x, y$, czas, ID, prędkości).

### Wyciągnięte Informacje i Statystyki

1.  **Identyfikacja Pasów Ruchu**:
    *   Na podstawie punktów wjazdu pojazdów (krawędzie obrazu) zidentyfikowano **8 pasów wjazdowych**:
        *   2 pasy z lewej strony
        *   3 pasy z góry
        *   3 pasy z prawej strony
    *   Zastosowano algorytm klastryzacji (K-Means) do automatycznego wykrywania środków pasów.

2.  **Mapy Ciepła (Heatmaps)**:
    *   **Zagęszczenie ruchu**: Wizualizacja obszarów o największym natężeniu ruchu.
        ![Mapa zagęszczenia ruchu](PLACEHOLDER_DENSITY_HEATMAP)
    *   **Prędkości**: Mapa średnich prędkości w [m/s] nałożona na obraz skrzyżowania (przelicznik: $0.117$ m/px).
        ![Mapa średnich prędkości](PLACEHOLDER_SPEED_HEATMAP)

3.  **Dynamika Pojazdów**:
    *   **Deceleracja (Hamowanie)**: Przeanalizowano fazy ciągłego hamowania (przeanalizowano 798,356 punktów danych).
        *   Średnia deceleracja: **2.85 m/s²**
        *   Mediana deceleracji: **2.67 m/s²**
        *   Ostre hamowanie (95. percentyl): **5.54 m/s²**
        
        ![Wykres rozkładu deceleracji](PLACEHOLDER_DECELERATION_PLOT)

    *   **Akceleracja (Przyspieszanie)**: Wyznaczono rozkład przyspieszeń pojazdów (przeanalizowano 830,667 punktów danych).
        *   Średnia akceleracja: **2.49 m/s²**
        *   Mediana akceleracji: **2.36 m/s²**
        *   95. percentyl: **4.66 m/s²**

        ![Wykres rozkładu akceleracji](PLACEHOLDER_ACCELERATION_PLOT)

Dzięki tym analizom uzyskano parametry niezbędne do kalibracji modeli symulacyjnych (np. typowe wartości przyspieszeń i prędkości dla tego konkretnego skrzyżowania).

