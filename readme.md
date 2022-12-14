# Эквализация гистограммы
## Базовые понятия
Для начала определим, что такое эквализация гистограммы.
Эквализация гистограммы - метод повышения контрастности изображения, путем "растягивания" гистограммы таким образом, чтобы самые темные пиксели стали черными (яркость была равна нулю), а самые яркие пиксели стали белыми (яркость равнялась 255).
Для того, чтобы найти изображение с эквализированной гистограммой, имея только исходное изображение, необходимо применить к нему некое преобразование equalize, такое что:

```equalized_img[x,y] = equalize[img[x,y]]```, где

`equalized_img[x,y]` - яркость пикселя с координатами x, y на изображении с эквализованной гистограммой;

`img[x,y]` - яркость пикселя исходного изображения;

`equalize` - некоторое преобразование между яркостью исходного пикселя и "эквализированного".

## Алгоритм работы
В качестве базового алгоритма был выбран алгоритм эквализации на основе функции распределения вероятностей гистограммы.
Шаги алгоритма:
1. Вычисление функции распределения вероятности гистограммы:

```cdf[i] = cdf[i-1] + hist[i]```, где

`cdf[i]` - значение функции распределения вероятности гистограммы от яркости пикселя i;

`hist[i]` - количество пикселей с яркостью i.

2. Вычисление функции соответствия яркости пикселей исходного и эквализированного изображения:

```eMap[i] = (k*(cdf[i] - cdf_min))```, где

`eMap[i]` - значение функции соответствия яркости пикселей исходного и эквализированного изображения от яркости пикселя i;

`cdf_min` - минимальное ненулевое значение функции распределения вероятности;

`k = 255.0 / (height*width - cdf_min)` - масштабный коэффициент, где

`height, width` - высота и ширина изображения в пикселях.

## Реализованная архитектура

Была реализована простая архитектура в функциональном стиле.

Существует основной цикл, захватывающий и демонстрирующий изображение и выполняющий эквализацию, случае выставления соответствующего флага.

Реализованы  следующие функции:
1. Функция вычисления гистограммы:
```python
def calcHist(source):
    hist = [0 for i in range(256)]
    rows, cols = source.shape
    for x in range(rows):
        for y in range(cols):
            value = source[x, y]
            hist[value] += 1
    return hist
```
2. Функция вычисления функции распределения вероятностей гистограммы:
```python
def makeCDF(hist):
    cdf = [0 for i in range(len(hist))]
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    return cdf
```
3. Функция, выполняющая основное преобразование по описанному выше алгоритму:
```python
def equalizeHist(source):
    hist = calcHist(source)
    rows, cols = source.shape
    cdf = makeCDF(hist)
    size = rows, cols, 1
    eHist = [0 for i in range(256)]
    cdf_min = 0
    for i in range(256):
        if (cdf[i] != 0):
            cdf_min = cdf[i]
            break
    k = 255.0 / (rows*cols - cdf_min)
    for i in range(256):
        eHist[i] = (k*(cdf[i] - cdf_min))
    equalizedImg = np.zeros(size, dtype=np.uint8)
    for x in range(rows):
        for y in range(cols):
            equalizedImg[x, y] = eHist[source[x, y]]
    return equalizedImg
```
4. Функция построения изображения с гистограммой:
```python
def make_hist_image(src, hist_w, hist_h, hist_size):
    histRange = (0, 256) # the upper boundary is exclusive
    hist = cv.calcHist(src, [0], None, [hist_size], histRange)
    bin_w = int(round( hist_w/hist_size ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv.normalize(hist, hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    for i in range(1, hist_size):
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(hist[i-1]) ),
            ( bin_w*(i), hist_h - int(hist[i]) ),
            ( 255, 0, 0), thickness=1)
    return histImage
```

Функции на cpp реализованы по такой же логике, как и функции на Python.

## Оценка производительности и выводы

В таблице ниже представлено среднее время эквализации одного кадра реализациями на Python с/без OpenCV, и реализация на cpp без использования OpenCV

| Python w OpenCV | Python w/o OpenCV | C++     |
|-----------------|-------------------|---------|
| 1.02 ms         | 266.48 ms         | 10.3 ms |

Очевидно, что самой быстрой оказалась реализация на Python с использованием OpenCV, что обусловлено реализацией алгоритмов Python с использованием кода на C, лучшей оптимизацией вычислений.

Хорошо заметная разница между собственными реализациями алгоритма на Python и С++ обусловлена разумеется отличиями между самими языками. В первую очередь необходимо упомянуть, что Python - интерпретируемый язык, а С++ - компилируемый. Большая часть ресурсов "съедается" работой самого интерпретатора Python. С++ же может работать еще быстрее. Указанное выше время было получено со стандартной оптимизацией  `-01`, если скомпилировать код с флагом `-Ofast` время обработки одного кадра поднимается до 6.5 ms, что на 30-40% быстрее, чем с оптимизацией по умолчанию.

## Как запустить

* Необходимо создать папку `samples` в корне репозитория и поместить в нее видеоролик формата mp4 с именем `1.mp4`

* Для запуска реализации на Python с использованием OpenCV:

`python3 hist_equalizer_py/hist_equalizer.py`

* Для запуска реализации на Python без использования OpenCV:

`python3 hist_equalizer_py_wo_opencv/hist_equalizer.py`

* Для запуска реализации на С++ на Linux:

`cd hist_equalizer_cpp_wo_opencv`

`mkdir build && cd build`

`cmake ..`

`make`

`./hist_equalizer`

Во всех реализациях для включения эквализации необходимо нажать клавишу `e`, для выхода - `q`

## Список источников
1. https://en.wikipedia.org/wiki/Histogram_equalization
2. https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html