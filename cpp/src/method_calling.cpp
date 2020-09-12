#include <math.h>
#include <nlopt.hpp>
#include <iostream>
#include <typeinfo>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <ctime>
#include <map>
#include "cnpy.h"
#include <cmath>
#include "boost/multi_array.hpp"
// #include "linterp.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include <cstddef>
#include "xtensor/xadapt.hpp"



template <class print_type>
void print(print_type to_print){
    std::cout << to_print << std::endl;
}


class Test {
    public:
        int a;
        Test (int arg) {
            a = arg;
        }
        void printing(int b) {
            print("printing a:");
            print(a);
            print(b);
        }
};

typedef void (Test::*TestMemFn)(int);

void passing_test(Test *ptr, TestMemFn f) {
    print("in test func");
    (ptr->*f)(12);
}


int main(){
    print("testing");
    Test t(20);
    Test * ptr;
    ptr = &t;
    TestMemFn p = &Test::printing;
    passing_test(ptr, p);
    return 0;
}
