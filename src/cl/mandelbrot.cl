#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


__kernel void mandelbrot(__global float buffer[], unsigned int width, unsigned int height, float fromX, float fromY, float sizeX, float sizeY,
unsigned int iters, float threshold, int smoothing)
{
    // Узнать какой workItem выполняется в этом потоке поможет функция get_global_id
    // см. в документации https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // OpenCL Compiler -> Built-in Functions -> Work-Item Functions

    // P.S. в общем случае количество элементов для сложения может быть некратно размеру WorkGroup, тогда размер рабочего пространства округлен вверх от числа элементов до кратности на размер WorkGroup
    // и в таком случае если сделать обращение к массиву просто по индексу=get_global_id(0) будет undefined behaviour (вплоть до повисания ОС)
    // поэтому нужно либо дополнить массив данных длиной до кратности размеру рабочей группы,
    // либо сделать return в кернеле до обращения к данным в тех WorkItems, где get_global_id(0) выходит за границы данных (явной проверкой)

    const unsigned int index = get_global_id(0);
    const float threshold2 = threshold * threshold;


    if (index < width * height) {
        float log_2 = log(2.0f);
        float log_t = log(threshold);
        int i = index % width;
        int j = index / width;

        float x0 = fromX + (i + 0.5f) * sizeX / width;
        float y0 = fromY + (j + 0.5f) * sizeY / height;

        float x = x0;
        float y = y0;

        int iter = 0;
        for (; iter < iters; ++iter) {
            float xPrev = x;
            x = x * x - y * y + x0;
            y = 2.0f * xPrev * y + y0;
            if ((x * x + y * y) > threshold2) {
                break;
            }
        }

            float result = iter;
            if (smoothing && iter != iters) {

                result = result - log(log(sqrt(x * x + y * y)) / log_t) / log_2;
            }

            result = 1.0f * result / iters;
            //results[j * width + i] = result;


        buffer[index] = result; //as[index] + bs[index];
    }
}


/*
__kernel void mandelbrot(...)
{
    // TODO если хочется избавиться от зернистости и дрожжания при интерактивном погружении - добавьте anti-aliasing:
    // грубо говоря при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
}
*/

