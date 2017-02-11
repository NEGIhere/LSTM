
#ifndef NNLIFE_UTILS_H
#define NNLIFE_UTILS_H

#include <stdlib.h>
#include <vector>
#include <cmath>
#include <iterator>

#define PI 3.14159265359f
#define rad2deg (1 / PI * 180.0f)
#define deg2rad (1 / 180.0f * PI)

class Utils {
public:
    static double randDouble(double first, double last) {
        return first + (rand() / double(RAND_MAX)) * (last - first);
    }

    static int randInt(int first, int last) {
        return first + (rand() % (last - first + 1));
    }

    inline float radToDeg(float rad) {
        return rad / PI * 180.f;
    }

    inline float degToRad(float deg) {
        return deg / 180.0f * PI;
    }

    static std::vector<double> softmax(const std::vector<double>& vec);

    template<typename T> static void print(const std::vector<T>& vec) {
        if (!vec.empty()) {
            std::cout << '[';
            //std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(std::cout, ", "));
            for (auto& it : vec) {
                printf("%.16f,", it);
            }
            std::cout << "\b\b]" << std::endl;
        }
    }
/*
    template <typename T> std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
        if ( !v.empty() ) {
            out << '[';
            std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
            out << "\b\b]";
        }
        return out;
    }*/
};

#endif //NNLIFE_UTILS_H
