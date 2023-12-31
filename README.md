В данной работе исследуются разбиения натурального числа $n$, части которого различны, нечетны и их произведение не является квадратом. Такие разбиения применимы для определения ранга группы центральных единиц целочисленного группового кольца знакопеременной группы.

Предложен параллельный алгоритм в общей памяти для нахождения количества разбиений числа $n$ с дополнительными условиями (см. all_omp и r4_fast_omp). Алгоритм основан на концепции распараллеливания по данным и использования вложенного параллелизма. Выделяется множество длин $K$ разбиения числа $n$, элементы которого обрабатываются параллельно. Во время обработки длины $k$ разбиения числа $n$ выделяется множество уровней $L$, рассмотрение которого также выполняется параллельно. Приемлемые значения ускорения и параллельной эффективности предложенного алгоритма получены при использования двух нитей на параллельный регион по длинам и двух --- по уровням. Таким образом, ускорение при разных $n$ превышает $2.1$, а параллельная эффективность не опускается ниже $50$% (см. charts/time.ipynb).

Предложен алгоритм поиска оптимального коэффициента $c$ (см. optimize). С помощью этого алгоритма получена асимптотическая формула количества разбиения числа $n$, в котором части различны и нечетны, а их произведение является квадратом (см. charts/charts.ipynb).

TODO: добавить BibTeX-ссылку на публикацию
