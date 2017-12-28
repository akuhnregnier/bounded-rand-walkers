#include "rejection_sampling.h"


int main() {
    struct timeval time;
    gettimeofday(&time, NULL);
    // from
    // https://stackoverflow.com/questions/322938/recommended-way-to-initialize-srand
    // microsecond has 1 000 000
    // Assuming you did not need quite that accuracy
    // Also do not assume the system clock has that accuracy.
    srand((time.tv_sec * 1000) + (time.tv_usec / 1000));
    // The trouble here is that the seed will repeat every
    // 24 days or so.

    // If you use 100 (rather than 1000) the seed repeats every 248 days.

    // Do not make the MISTAKE of using just the tv_usec
    // This will mean your seed repeats every second.

    // testing_1d();
    testing_2d_redux();
    // testing_2d();
    // test_polygon();
}
