
#ifndef NNLIFE_UTILS_H
#define NNLIFE_UTILS_H

#include <stdlib.h>

#define PI 3.14159265359f
#define rad2deg (1 / PI * 180.0f)
#define deg2rad (1 / 180.0f * PI)

class Utils {
public:
    static double randomDouble(double first, double last) {
        return first + (rand() / double(RAND_MAX)) * (last - first);
    }

    static int randomInt(int first, int last) {
        return first + (rand() % (last - first + 1));
    }

    inline float radToDeg(float rad) {
        return rad / PI * 180.f;
    }

    inline float degToRad(float deg) {
        return deg / 180.0f * PI;
    }
};


#endif //NNLIFE_UTILS_H
